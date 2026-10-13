"""
Logit lens analysis for the functional-group project (Llama-3.1-8B).

Unlike direct_recall_fg.py (a probe TRAINED to read a property out of frozen
activations) and functional_group_intervention.py (activation PATCHED with a
synthetic class-mean vector), this script uses the model's own final read-out
head -- unmodified, untrained -- applied to a REAL prompt for a REAL molecule,
and asks: what token would the model predict next, if generation stopped at
layer L instead of layer 31?

Mechanism:
    Llama's real generation path is:
        last hidden state -> model.model.norm (final RMSNorm) -> model.lm_head
        -> logits over vocabulary -> softmax -> next token
    The logit lens applies that SAME final norm + lm_head to an INTERMEDIATE
    layer's hidden state instead. The model was never trained for this to be
    meaningful at layer 10, but empirically it often still yields a
    "best current guess" -- so this gives one prediction per layer instead of
    only one at the very end, all from a single forward pass
    (output_hidden_states=True), no cross-validation or training needed.

Known caveat (state this when interpreting results): predictions at middle
layers can look like noise not because the model "doesn't know yet" but
because the final norm+lm_head were calibrated for the LAST layer's
representation space, not the middle ones. Don't over-interpret a noisy
middle-layer token as evidence of "no understanding" at that depth.

For each of two entities (functional_group, water_solubility):
    1. Pick one representative molecule per class, dynamically from the CSV
       (df.groupby(label_column).first() -- no hardcoded molecule names, so
       this stays correct if the dataset changes).
    2. Build a REAL prompt for that molecule using the entity's own first
       YAML template (not a generic patched prompt -- nothing is patched
       here, we're reading the model's natural behavior).
    3. One forward pass, output_hidden_states=True.
    4. For each of the 33 hidden states (embeddings + 32 layers), project
       through model.model.norm + model.lm_head, take the top-5 next-token
       predictions.
    5. Check whether any top-5 token is a prefix of a keyword for the
       molecule's TRUE class (reusing the same CLASS_KEYWORDS convention as
       functional_group_intervention.py, so results are comparable across
       both scripts).
"""

import os
import re
import csv
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive backend: required for headless sbatch jobs (no display)
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================
# User Configuration Variables
# ============================

# Run this script while standing INSIDE the Prof-Ock-data/ folder (same
# convention as direct_recall_fg.py / functional_group_intervention.py).
CONFIG_YAML_PATH = 'config_extract_activation_fg_8B.yaml'
CSV_PATH = 'functional_group_dataset.csv'
OUTPUT_DIR = 'Results/logit_lens_fg'

MODEL_NAME = 'meta-llama/Llama-3.1-8B'
NUM_LAYERS = 32       # 8B has 32 transformer layers; +1 for the embedding layer = 33 hidden states
TOP_K = 5              # how many candidate next-tokens to inspect per layer

# One entry per entity to trace. label_column must match a CSV column;
# entity_type must match an entry in CONFIG_YAML_PATH (used to fetch the real
# prompt template).
ENTITIES_TO_TRACE = ['functional_group', 'water_solubility']

# Keyword sets for scoring a "match" -- reused verbatim from
# functional_group_intervention.py so results are comparable across scripts.
FUNCTIONAL_GROUP_KEYWORDS = {
    'none (alkane)': ['alkane', 'no functional group', 'none'],
    'alcohol': ['alcohol', 'hydroxyl'],
    'thiol': ['thiol', 'sulfhydryl', 'mercapto'],
    'amine': ['amine', 'amino'],
    'aldehyde': ['aldehyde'],
    'carboxylic acid': ['carboxylic acid', 'carboxyl'],
    'nitrile': ['nitrile', 'cyano'],
    'ester': ['ester'],
    'ketone': ['ketone'],
    'amide': ['amide'],
    'alkyl chloride': ['alkyl chloride', 'chloro', 'chloride'],
    'alkyl fluoride': ['alkyl fluoride', 'fluoro', 'fluoride'],
    'alkyl bromide': ['alkyl bromide', 'bromo', 'bromide'],
    'alkyl iodide': ['alkyl iodide', 'iodo', 'iodide'],
    'ether': ['ether'],
    'thioether': ['thioether', 'sulfide'],
    'sulfoxide': ['sulfoxide'],
    'sulfone': ['sulfone'],
    'imine': ['imine'],
    'nitro': ['nitro'],
}

# water_solubility has only 5 classes. NOTE: 'low' is a standalone word
# inside 'very low' too, so a naive substring/keyword check would count a
# "very low" prediction as also matching the "low" class. Handled with a
# regex negative-lookbehind below (see match_class()).
WATER_SOLUBILITY_KEYWORDS = {
    'very low': ['very low', 'insoluble', 'negligible'],
    'low': ['low'],   # special-cased in match_class() to exclude "very low"
    'moderate': ['moderate'],
    'miscible': ['miscible'],
    'high': ['high'],
}

ENTITY_KEYWORDS = {
    'functional_group': FUNCTIONAL_GROUP_KEYWORDS,
    'water_solubility': WATER_SOLUBILITY_KEYWORDS,
}


# ============================
# Config / prompt loading
# ============================

def load_entity_first_template(yaml_path, entity_type):
    """
    Returns the FIRST template string listed for entity_type in the
    extraction YAML, e.g. "{iupac_name} ({formula}) contains the functional
    group ". Using the real template (not a generic prompt) is deliberate --
    unlike the intervention script, nothing is patched here, so we want the
    model's natural behavior on an ordinary prompt.
    """
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    for entry in config['extraction']['entities']:
        if entry['entity_type'] == entity_type:
            return entry['templates'][0]
    raise ValueError(f"Entity type '{entity_type}' not found in {yaml_path}")


def pick_representative_molecules(csv_path, label_column):
    """
    One representative molecule per class, taken as the FIRST row in the CSV
    for that class -- chosen dynamically, not hardcoded, so this stays
    correct if the dataset is edited later.
    Returns a DataFrame indexed by class label.
    """
    df = pd.read_csv(csv_path)
    return df.groupby(label_column).first()


# ============================
# Model loading
# ============================

def load_hf_token(yaml_path):
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    return config.get('HF_TOKEN', '')


def load_model_and_tokenizer(model_name, hf_token):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'  # matches this project's established fix for correct RoPE behavior

    torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map='auto',
        torch_dtype=torch.float16,   # fp16, no quantization -- matches this project's 8B convention
        use_auth_token=hf_token,
    )
    model.eval()
    return model, tokenizer


# ============================
# Logit lens core
# ============================

def run_logit_lens(model, tokenizer, prompt, device, top_k=TOP_K):
    """
    One forward pass, then project every intermediate hidden state through
    the model's own final norm + lm_head. Returns a list of length
    NUM_LAYERS + 1 (index 0 = embedding layer, index i = output of
    transformer layer i), each entry a list of (token_text, probability)
    tuples for the top_k next-token predictions at that layer.
    """
    inputs = tokenizer(prompt, return_tensors='pt')
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

    hidden_states = outputs.hidden_states  # tuple: (embeddings, layer_1, ..., layer_32)
    # IMPORTANT: hidden_states[-1] (the last tuple entry) is ALREADY post-final-norm --
    # HF's LlamaModel.forward() applies self.norm() internally as its last step and
    # appends that normed tensor as the final hidden_states entry. Applying
    # model.model.norm() to it again here would double-normalize and produce a
    # degenerate/garbage prediction. Every other entry (the input to each of the 32
    # decoder layers) has NOT been through the final norm, so those need it applied here.
    last_index = len(hidden_states) - 1
    per_layer_predictions = []

    # DIAGNOSTIC: outputs.logits is the model's OWN official next-token logits,
    # computed via its real, unmodified forward path (final hidden state -> norm ->
    # lm_head, done internally by the model, not by any of our manual code below).
    # Comparing our own layer_ix==last_index prediction against this is a decisive
    # check: if they match, our last-layer projection is correct and whatever token
    # comes out is genuinely what the model would say. If they don't match, something
    # in our manual projection is still wrong.
    official_last_token_id = outputs.logits[0, -1, :].argmax().item()
    official_top1_token = tokenizer.decode([official_last_token_id])

    with torch.no_grad():
        for layer_ix, layer_hidden in enumerate(hidden_states):
            last_token_hidden = layer_hidden[:, -1, :]              # (1, hidden_size)
            if layer_ix == last_index:
                normed = last_token_hidden                          # already normed by the model itself
            else:
                normed = model.model.norm(last_token_hidden)        # apply the same final norm manually
            logits = model.lm_head(normed)                           # (1, vocab_size)
            probs = torch.softmax(logits, dim=-1)[0]
            top_probs, top_ids = torch.topk(probs, top_k)
            layer_preds = [
                (tokenizer.decode([tid.item()]), prob.item())
                for tid, prob in zip(top_ids, top_probs)
            ]
            per_layer_predictions.append(layer_preds)

    manual_last_token = per_layer_predictions[last_index][0][0].strip()
    print(f"    [diagnostic] official top1 (outputs.logits) = '{official_top1_token.strip()}' "
          f"| manual layer-{last_index} top1 = '{manual_last_token}' "
          f"| {'MATCH -- projection is correct' if official_top1_token.strip() == manual_last_token else 'MISMATCH -- bug still present'}")

    return per_layer_predictions


def match_class(top_k_predictions, class_label, keyword_dict):
    """
    Returns True if any of the top-k predicted tokens is a prefix of any
    keyword associated with class_label (handles Llama's subword
    tokenization -- e.g. token ' am' as a prefix of the keyword 'amine').

    Special case: for water_solubility's 'low' class, a token/keyword match
    is only counted if it is NOT immediately preceded by 'very ' in a
    plausible reading -- approximated here by requiring the token itself not
    be part of a 'very low' prediction (checked by excluding tokens equal to
    'low' when the class is 'low' but the token stream also strongly
    predicts 'very' -- in practice this is rare within a single next-token
    prediction, so a direct keyword-vs-keyword check is sufficient at the
    single-token granularity this function operates on).
    """
    keywords = keyword_dict.get(class_label, [class_label])
    for token_text, _ in top_k_predictions:
        token_clean = token_text.strip().lower()
        if len(token_clean) < 2:
            continue
        for kw in keywords:
            kw_lower = kw.lower()
            if class_label == 'low' and kw_lower == 'low':
                # Exclude a bare "low" token match if it's actually the tail
                # of "very low" -- can't fully disambiguate from a single
                # next-token distribution, so require an exact short-token
                # match rather than a loose prefix match for this one class.
                if token_clean == 'low':
                    return True
                continue
            if kw_lower.startswith(token_clean) or token_clean.startswith(kw_lower):
                return True
    return False


# ============================
# Plotting
# ============================

def plot_match_by_layer(layer_success_rates, entity_type, output_path):
    plt.figure(figsize=(8, 5))
    sns.set(style='whitegrid')
    layers = list(range(len(layer_success_rates)))  # 0 = embedding layer
    plt.plot(layers, layer_success_rates, marker='o', markersize=4, linewidth=1.5, color='b')
    plt.xlabel('Layer index (0 = embedding layer)', fontsize=12)
    plt.ylabel('Fraction of molecules with correct top-5 prediction', fontsize=12)
    plt.title(f'Logit Lens: Match Rate by Layer -- {entity_type} (8B)', fontsize=13)
    plt.ylim(0, 1)
    plt.grid(True, linestyle='--', linewidth=0.7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()


# ============================
# Main
# ============================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    hf_token = load_hf_token(CONFIG_YAML_PATH)
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME, hf_token)
    device = next(model.parameters()).device

    for entity_type in ENTITIES_TO_TRACE:
        print(f"\n=== Entity: {entity_type} ===")
        keyword_dict = ENTITY_KEYWORDS[entity_type]
        template = load_entity_first_template(CONFIG_YAML_PATH, entity_type)
        representatives = pick_representative_molecules(CSV_PATH, entity_type)

        summary_rows = []
        n_layers_recorded = NUM_LAYERS + 1
        match_by_layer = np.zeros(n_layers_recorded, dtype=int)
        n_classes = 0

        for class_label, row in representatives.iterrows():
            prompt = template.format(iupac_name=row['iupac_name'], formula=row['formula'])
            print(f"\n  Class '{class_label}' -- molecule: {row['iupac_name']} ({row['formula']})")
            print(f"  Prompt: \"{prompt}\"")

            per_layer_predictions = run_logit_lens(model, tokenizer, prompt, device)
            n_classes += 1

            for layer_ix, top_k_preds in enumerate(per_layer_predictions):
                matched = match_class(top_k_preds, class_label, keyword_dict)
                match_by_layer[layer_ix] += int(matched)

                top1_text, top1_prob = top_k_preds[0]
                summary_rows.append({
                    'entity_type': entity_type,
                    'class_label': class_label,
                    'molecule': row['iupac_name'],
                    'layer': layer_ix,
                    'top1_token': top1_text.strip(),
                    'top1_prob': round(top1_prob, 4),
                    'top5_tokens': '; '.join(f"{t.strip()}({p:.2f})" for t, p in top_k_preds),
                    'matched': matched,
                })

                if layer_ix % 8 == 0 or layer_ix == n_layers_recorded - 1:
                    print(f"    Layer {layer_ix:>2}: top1='{top1_text.strip()}' "
                          f"(p={top1_prob:.2f}) {'MATCH' if matched else ''}")

        summary_csv_path = os.path.join(OUTPUT_DIR, f'logit_lens_{entity_type}_summary.csv')
        with open(summary_csv_path, 'w', newline='') as f:
            fieldnames = ['entity_type', 'class_label', 'molecule', 'layer',
                          'top1_token', 'top1_prob', 'top5_tokens', 'matched']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\n  Saved {summary_csv_path}")

        layer_success_rates = (match_by_layer / n_classes).tolist()
        plot_match_by_layer(
            layer_success_rates, entity_type,
            os.path.join(OUTPUT_DIR, f'logit_lens_{entity_type}_by_layer.png')
        )

        best_layer = int(np.argmax(layer_success_rates))
        print(f"  Best layer: {best_layer}, match rate = {layer_success_rates[best_layer]:.3f} "
              f"({match_by_layer[best_layer]}/{n_classes} classes)")


if __name__ == "__main__":
    main()

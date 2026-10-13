"""
Bare-name / open-question logit lens for the functional-group project
(Llama-3.1-8B).

Every other logit-lens experiment so far used a prompt that names a SPECIFIC
property ("...contains the functional group ", "...has an approximate pKa
of ") -- this steers the model toward a specific kind of answer by
construction, so it's hard to tell whether a concept token (e.g. "acid",
"halo") shows up because the model spontaneously associates it with the
molecule, or merely because the prompt's grammar demands that kind of
continuation.

This script tests two prompt styles that remove that property-specific
steering, at two different levels:
  - 'bare': just the molecule's own name, no question at all. Round 1
    (1-Chlorohexane) found this drifts toward generic database-entry
    formatting (CAS, punctuation) rather than any chemistry content --
    likely too little structure for the model to know what kind of
    continuation is expected.
  - 'what_is': "What is {name}?" -- a genuine question, so the model knows
    it should produce a substantive description, but WITHOUT specifying
    which property to describe. This is the middle ground between 'bare'
    and the fully property-specific prompts used elsewhere: it should
    reveal what the model considers most salient/important to say about a
    molecule when given free rein, rather than being asked for one
    particular fact.

No "matched" scoring here (there's no single correct answer to either
prompt style) -- this is purely exploratory. Saves every layer's top-5
predictions for manual reading, same format as the other logit-lens
scripts so results are easy to compare side by side, and directly
comparable between the two prompt styles for the same molecule.
"""

import os
import csv
import yaml
import torch
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive backend: required for headless sbatch jobs (no display)

from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================
# User Configuration Variables
# ============================

CONFIG_YAML_PATH = 'config_extract_activation_fg_8B.yaml'
CSV_PATH = 'functional_group_dataset.csv'
OUTPUT_DIR = 'Results/logit_lens_fg'

MODEL_NAME = 'meta-llama/Llama-3.1-8B'
NUM_LAYERS = 32
TOP_K = 5

# Round 2: more common/recognizable molecules than the pilot (1-Chlorohexane
# was a fairly obscure, systematic-only compound). Includes a real amine
# (Butan-1-amine) specifically to check whether "amine" appears correctly
# here, given it showed up as a confusing false positive for the (non-amine)
# 1-Chlorohexane pilot.
MOLECULES_TO_TEST = [
    'Propan-2-one',      # Acetone -- well-known solvent (ketone)
    'Propanoic acid',    # close to acetic acid / vinegar (carboxylic acid)
    'Propan-1-ol',       # propanol (alcohol)
    'Methyl ethanoate',  # methyl acetate -- nail-polish-remover component (ester)
    'Butan-1-amine',     # real amine -- control for the earlier false "amine" hit
    '1-Nitropropane',    # less famous, kept as a contrast/control
]

# Each molecule is tested with BOTH prompt styles below, so the two can be
# compared side by side for the same molecule in the same run.
PROMPT_STYLES = {
    'bare': '{name}',
    'what_is': 'What is {name}?',
}


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
    tokenizer.padding_side = 'right'

    torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map='auto',
        torch_dtype=torch.float16,
        use_auth_token=hf_token,
    )
    model.eval()
    return model, tokenizer


# ============================
# Logit lens core (identical mechanism to logit_lens_fg.py, including the
# fix for hidden_states[-1] already being post-final-norm)
# ============================

def run_logit_lens(model, tokenizer, prompt, device, top_k=TOP_K):
    inputs = tokenizer(prompt, return_tensors='pt')
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

    hidden_states = outputs.hidden_states
    last_index = len(hidden_states) - 1  # already post-final-norm
    per_layer_predictions = []

    with torch.no_grad():
        for layer_ix, layer_hidden in enumerate(hidden_states):
            last_token_hidden = layer_hidden[:, -1, :]
            normed = last_token_hidden if layer_ix == last_index else model.model.norm(last_token_hidden)
            logits = model.lm_head(normed)
            probs = torch.softmax(logits, dim=-1)[0]
            top_probs, top_ids = torch.topk(probs, top_k)
            per_layer_predictions.append([
                (tokenizer.decode([tid.item()]), prob.item())
                for tid, prob in zip(top_ids, top_probs)
            ])

    return per_layer_predictions


# ============================
# Main
# ============================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    hf_token = load_hf_token(CONFIG_YAML_PATH)
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME, hf_token)
    device = next(model.parameters()).device

    df = pd.read_csv(CSV_PATH)
    n_layers_recorded = NUM_LAYERS + 1
    summary_rows = []

    for name in MOLECULES_TO_TEST:
        match = df[df['iupac_name'] == name]
        if match.empty:
            print(f"WARNING: '{name}' not found in {CSV_PATH}, skipping.")
            continue
        row = match.iloc[0]
        fg_class = row['functional_group']

        for style_name, style_template in PROMPT_STYLES.items():
            prompt = style_template.format(name=name)

            print(f"\n=== {name} [true functional_group = {fg_class}] -- style: {style_name} ===")
            print(f"Prompt (verbatim): \"{prompt}\"")

            per_layer_predictions = run_logit_lens(model, tokenizer, prompt, device)

            for layer_ix, top_k_preds in enumerate(per_layer_predictions):
                top1_text, top1_prob = top_k_preds[0]
                summary_rows.append({
                    'molecule': name,
                    'fg_class': fg_class,
                    'prompt_style': style_name,
                    'layer': layer_ix,
                    'top1_token': top1_text.strip(),
                    'top1_prob': round(top1_prob, 4),
                    'top5_tokens': '; '.join(f"{t.strip()}({p:.2f})" for t, p in top_k_preds),
                })
                print(f"  Layer {layer_ix:>2}: {'; '.join(f'{t.strip()!r}({p:.2f})' for t, p in top_k_preds)}")

    if not summary_rows:
        print("\nNo molecules were processed -- check MOLECULES_TO_TEST against the CSV.")
        return

    summary_csv_path = os.path.join(OUTPUT_DIR, 'logit_lens_bare_name_summary.csv')
    with open(summary_csv_path, 'w', newline='') as f:
        fieldnames = ['molecule', 'fg_class', 'prompt_style', 'layer', 'top1_token', 'top1_prob', 'top5_tokens']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nSaved {summary_csv_path}")


if __name__ == "__main__":
    main()

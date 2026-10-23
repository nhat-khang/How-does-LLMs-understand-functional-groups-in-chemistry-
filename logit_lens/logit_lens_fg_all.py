"""
Combined logit-lens sweep across 5 entities for the functional-group project
(Llama-3.1-8B): functional_group, water_solubility (categorical), and
pka, pkah, mw (continuous).

Design choices, and why:

1. ALL 92 molecules, for all 5 entities (not just one representative per
   functional_group class). Rows are sorted by (functional_group,
   iupac_name) so every molecule of the same class sits adjacent on the
   heatmap -- this lets Anh check whether a class's match/no-match pattern
   holds across ALL its molecules (real, class-level effect) or is specific
   to a single molecule's name length/frequency (an artifact unrelated to
   chemistry). Every entity's prompt is built for the SAME 92 molecules, so
   the only thing that differs between entities is the PROMPT, never the
   molecule sample.

2. Two different "matched" criteria, by entity kind:
   - Categorical (functional_group, water_solubility): reuses the existing
     keyword-prefix matching from logit_lens_fg.py (CLASS_KEYWORDS /
     WATER_SOLUBILITY_KEYWORDS), checked against the top-5 predictions.
   - Continuous (pka, pkah, mw): a prediction counts as matched if ANY of
     the top-5 tokens parses as a number within NUMERIC_MATCH_TOLERANCE
     (default 20%) relative error of the molecule's true value. Relative
     (not absolute) tolerance is used because pKa (~4-50) and mw (~44-200)
     have very different natural scales -- a single absolute tolerance
     would be too loose for one and too strict for the other.
   Rows where the true value is missing (NaN -- e.g. pkah/pka for several
   alkanes/alkyl halides, water_solubility for some molecules) are left
   unmatched (shown as a distinct "no ground truth" color in the heatmap,
   not counted as False).

3. One heatmap PER ENTITY (5 images total), not one combined image -- 5
   entities x 92 molecules x 33 layers in a single figure would be
   unreadable. Each heatmap: rows = individual molecule, grouped by
   functional-group class with a separator line between classes (same
   molecule order across all 5, for direct comparison), columns = layer
   0-32, cell = matched (dark), not matched (light), no ground truth (gray).

4. The CSV keeps top1_token AND the full top5_tokens string with
   probabilities for every (entity, molecule, layer) row, so the middle
   layers' actual predicted words/numbers can be read directly -- this is
   what surfaced the "pH" / "acid" concept-stage tokens for pka/pkah before
   the model committed to an actual number in the final layers.
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

CONFIG_YAML_PATH = 'config_extract_activation_fg_8B.yaml'
CSV_PATH = 'functional_group_dataset.csv'
OUTPUT_DIR = 'Results/logit_lens_fg'

MODEL_NAME = 'meta-llama/Llama-3.1-8B'
NUM_LAYERS = 32
TOP_K = 5

NUMERIC_MATCH_TOLERANCE = 0.20  # relative error threshold for pka/pkah/mw

# Suffix added to every output filename, so a run with a different scope
# (e.g. back to 20 representatives, or a different molecule subset) doesn't
# silently overwrite this one. Change this before changing what the script
# actually runs on.
RUN_LABEL = '92mol'

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

WATER_SOLUBILITY_KEYWORDS = {
    'very low': ['very low', 'insoluble', 'negligible'],
    'low': ['low'],   # special-cased below to exclude "very low"
    'moderate': ['moderate'],
    'miscible': ['miscible'],
    'high': ['high'],
}

# entity_type -> config: kind ('categorical' or 'numeric'), the CSV column
# holding its true label/value, and (for categorical) its keyword dict.
ENTITY_CONFIG = {
    'functional_group': {'kind': 'categorical', 'label_column': 'functional_group', 'keywords': FUNCTIONAL_GROUP_KEYWORDS},
    'water_solubility': {'kind': 'categorical', 'label_column': 'water_solubility', 'keywords': WATER_SOLUBILITY_KEYWORDS},
    'pka': {'kind': 'numeric', 'label_column': 'pka'},
    'pkah': {'kind': 'numeric', 'label_column': 'pkah'},
    'mw': {'kind': 'numeric', 'label_column': 'mw'},
}
ENTITIES_TO_RUN = ['functional_group', 'water_solubility', 'pka', 'pkah', 'mw']


# ============================
# Config / prompt loading
# ============================

def load_entity_first_template(yaml_path, entity_type):
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    for entry in config['extraction']['entities']:
        if entry['entity_type'] == entity_type:
            return entry['templates'][0]
    raise ValueError(f"Entity type '{entity_type}' not found in {yaml_path}")


def load_all_molecules_sorted(csv_path, class_column='functional_group', name_column='iupac_name'):
    """
    Every molecule in the dataset, sorted by (class_column, name_column) so
    molecules of the same functional-group class are adjacent rows -- makes
    the heatmap directly show whether a class's match pattern is consistent
    across all its molecules or specific to just one.
    functional_group stays a REGULAR column here (unlike the old
    groupby().first() approach, which turned it into the index and caused a
    KeyError when later looking it up as representatives['functional_group']).
    """
    df = pd.read_csv(csv_path)
    return df.sort_values([class_column, name_column]).reset_index(drop=True)


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
# Logit lens core
# ============================

def run_logit_lens(model, tokenizer, prompt, device, top_k=TOP_K):
    inputs = tokenizer(prompt, return_tensors='pt')
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

    hidden_states = outputs.hidden_states
    last_index = len(hidden_states) - 1  # already post-final-norm; see logit_lens_fg.py for why
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


def match_categorical(top_k_predictions, class_label, keyword_dict):
    keywords = keyword_dict.get(class_label, [class_label])
    for token_text, _ in top_k_predictions:
        token_clean = token_text.strip().lower()
        if len(token_clean) < 2:
            continue
        for kw in keywords:
            kw_lower = kw.lower()
            if class_label == 'low' and kw_lower == 'low':
                if token_clean == 'low':
                    return True
                continue
            if kw_lower.startswith(token_clean) or token_clean.startswith(kw_lower):
                return True
    return False


def parse_number(token_text):
    text = token_text.strip()
    try:
        return float(text)
    except ValueError:
        m = re.match(r'^-?\d+\.?\d*', text)
        return float(m.group()) if m else None


def match_numeric(top_k_predictions, true_value, tolerance=NUMERIC_MATCH_TOLERANCE):
    """
    Returns True/False, or None if true_value is missing (no ground truth
    to compare against -- e.g. pkah undefined for alkanes).
    """
    if true_value is None or (isinstance(true_value, float) and np.isnan(true_value)):
        return None
    for token_text, _ in top_k_predictions:
        pred = parse_number(token_text)
        if pred is None:
            continue
        if true_value == 0:
            if abs(pred) < 1e-6:
                return True
            continue
        if abs(pred - true_value) / abs(true_value) <= tolerance:
            return True
    return False


# ============================
# Plotting
# ============================

def plot_match_heatmap(match_matrix, row_labels, class_boundaries, entity_type, output_path):
    """
    match_matrix: 2D array (n_molecules x n_layers), values in {0, 1, np.nan}
    (np.nan = no ground truth available for that molecule/entity).
    row_labels: molecule name per row (iupac_name), same order as match_matrix.
    class_boundaries: row indices where a new functional_group class starts
    (besides row 0) -- a horizontal line is drawn at each, so blocks of
    same-class molecules are visually separated.
    """
    n_rows = len(row_labels)
    cmap = sns.color_palette(['#f0f0f0', '#08519c'])  # light gray (False) -> dark blue (True)
    plt.figure(figsize=(12, max(8, 0.22 * n_rows)))
    ax = sns.heatmap(
        match_matrix, cmap=cmap, vmin=0, vmax=1,
        mask=np.isnan(match_matrix),
        cbar_kws={'label': 'Matched (dark) / Not matched (light)', 'ticks': [0, 1]},
        yticklabels=row_labels, xticklabels=range(match_matrix.shape[1]),
        linewidths=0.2, linecolor='white',
    )
    ax.set_facecolor('#d9d9d9')  # gray background shows through masked (no-ground-truth) cells
    for boundary in class_boundaries:
        ax.axhline(boundary, color='black', linewidth=1.2)
    plt.xlabel('Layer index (0 = embedding layer)')
    plt.ylabel('Molecule (grouped by functional-group class, separated by black lines)')
    plt.title(f'Logit Lens Match Heatmap -- {entity_type} (8B), all 92 molecules')
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

    molecules = load_all_molecules_sorted(CSV_PATH)
    row_labels = molecules['iupac_name'].tolist()
    fg_classes = molecules['functional_group'].tolist()
    # A boundary line is drawn wherever the functional_group class changes
    # from the previous row (skip row 0, nothing to separate it from).
    class_boundaries = [i for i in range(1, len(fg_classes)) if fg_classes[i] != fg_classes[i - 1]]
    print(f"Using all {len(molecules)} molecules, sorted by functional_group class, "
          f"for all {len(ENTITIES_TO_RUN)} entities.")

    n_layers_recorded = NUM_LAYERS + 1

    for entity_type in ENTITIES_TO_RUN:
        cfg = ENTITY_CONFIG[entity_type]
        print(f"\n=== Entity: {entity_type} ({cfg['kind']}) ===")
        template = load_entity_first_template(CONFIG_YAML_PATH, entity_type)

        # Numeric columns need coercion (pkah mixes real numbers with
        # explanatory text for rows where the property doesn't apply).
        if cfg['kind'] == 'numeric':
            true_values_col = pd.to_numeric(molecules[cfg['label_column']], errors='coerce')

        summary_rows = []
        match_matrix = np.full((len(molecules), n_layers_recorded), np.nan)

        for row_ix, row in molecules.iterrows():
            fg_class = row['functional_group']
            prompt = template.format(iupac_name=row['iupac_name'], formula=row['formula'])
            true_value = true_values_col.iloc[row_ix] if cfg['kind'] == 'numeric' else row[cfg['label_column']]
            print(f"\n  [{fg_class}] {row['iupac_name']} ({row['formula']}) -- true {entity_type} = {true_value}")

            per_layer_predictions = run_logit_lens(model, tokenizer, prompt, device)

            for layer_ix, top_k_preds in enumerate(per_layer_predictions):
                if cfg['kind'] == 'categorical':
                    if true_value is None or (isinstance(true_value, float) and np.isnan(true_value)):
                        matched = None  # no ground truth for this molecule -- e.g. missing water_solubility
                    else:
                        matched = match_categorical(top_k_preds, true_value, cfg['keywords'])
                else:
                    matched = match_numeric(top_k_preds, true_value)

                match_matrix[row_ix, layer_ix] = np.nan if matched is None else float(matched)

                top1_text, top1_prob = top_k_preds[0]
                summary_rows.append({
                    'entity_type': entity_type,
                    'fg_class': fg_class,
                    'molecule': row['iupac_name'],
                    'true_value': true_value,
                    'layer': layer_ix,
                    'top1_token': top1_text.strip(),
                    'top1_prob': round(top1_prob, 4),
                    'top5_tokens': '; '.join(f"{t.strip()}({p:.2f})" for t, p in top_k_preds),
                    'matched': matched,
                })

        summary_csv_path = os.path.join(OUTPUT_DIR, f'logit_lens_{entity_type}_summary_{RUN_LABEL}.csv')
        with open(summary_csv_path, 'w', newline='') as f:
            fieldnames = ['entity_type', 'fg_class', 'molecule', 'true_value', 'layer',
                          'top1_token', 'top1_prob', 'top5_tokens', 'matched']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\n  Saved {summary_csv_path}")

        heatmap_path = os.path.join(OUTPUT_DIR, f'logit_lens_{entity_type}_heatmap_{RUN_LABEL}.png')
        plot_match_heatmap(match_matrix, row_labels, class_boundaries, entity_type, heatmap_path)
        print(f"  Saved {heatmap_path}")

        valid = ~np.isnan(match_matrix)
        if valid.any():
            rate_by_layer = np.nanmean(match_matrix, axis=0)
            best_layer = int(np.nanargmax(rate_by_layer))
            print(f"  Best layer: {best_layer}, match rate = {rate_by_layer[best_layer]:.3f}")


if __name__ == "__main__":
    main()

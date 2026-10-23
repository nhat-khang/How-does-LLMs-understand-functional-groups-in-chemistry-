"""
Cross-check for the functional-group project (Llama-3.1-8B): does the
degenerate "1"-token bias found in the functional_group logit lens run
generalize to other entities, or is it specific to that prompt?

Motivation: t-SNE found that pka/pkah/mw activations organize far more
cleanly by functional_group (silhouette 0.78 / 0.68 / 0.53) than
functional_group's OWN activations do (-0.04). One untested hypothesis for
why functional_group's logit lens got stuck predicting "1" at nearly every
layer: maybe greedy-decoding bias toward "1" is a general property of these
prompts, not something specific to functional_group. This script tests that
directly by running the same logit-lens procedure on pka, pkah, and mw.

Key design choice: reuses the EXACT SAME 20 representative molecules already
picked for functional_group (one per functional-group class), instead of
picking new representatives for each numeric entity. This keeps the
molecule sample fixed across the comparison, so any difference in the "1"
bias rate can only be attributed to the PROMPT (which property is being
asked about), not to a different set of molecules being tested.

Unlike the categorical logit_lens_fg.py, no keyword-matching is used here --
pka/pkah/mw are continuous, so there's no keyword to match. Instead this
just records the top-1/top-5 tokens per layer and computes, per entity, the
FRACTION OF LAYERS where the top-1 token is literally "1" -- directly
comparable to functional_group's own "1"-bias rate.
"""

import os
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
# convention as direct_recall_fg.py / functional_group_intervention.py /
# logit_lens_fg.py).
CONFIG_YAML_PATH = 'config_extract_activation_fg_8B.yaml'
CSV_PATH = 'functional_group_dataset.csv'
OUTPUT_DIR = 'Results/logit_lens_fg'

MODEL_NAME = 'meta-llama/Llama-3.1-8B'
NUM_LAYERS = 32
TOP_K = 5

# Continuous entities to cross-check, chosen because t-SNE found them to be
# the strongest INCIDENTAL organizers of functional_group (see Topic 2).
NUMERIC_ENTITIES_TO_TRACE = ['pka', 'pkah', 'mw']

# The class column used ONLY to pick the fixed set of 20 representative
# molecules -- same molecules as the functional_group logit-lens run, so the
# comparison varies only the prompt, not the molecule sample.
REPRESENTATIVE_CLASS_COLUMN = 'functional_group'


# ============================
# Config / prompt loading
# ============================

def load_entity_first_template(yaml_path, entity_type):
    """
    Returns the FIRST template string listed for entity_type in the
    extraction YAML. Same convention as logit_lens_fg.py -- a real,
    natural prompt, nothing patched or generic.
    """
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    for entry in config['extraction']['entities']:
        if entry['entity_type'] == entity_type:
            return entry['templates'][0]
    raise ValueError(f"Entity type '{entity_type}' not found in {yaml_path}")


def pick_fixed_representative_molecules(csv_path, class_column):
    """
    One representative molecule per class of REPRESENTATIVE_CLASS_COLUMN
    (functional_group), taken as the FIRST row per class -- IDENTICAL
    selection logic to logit_lens_fg.py's pick_representative_molecules(),
    so this is the exact same 20 molecules already used there.
    """
    df = pd.read_csv(csv_path)
    return df.groupby(class_column).first()


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
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

    hidden_states = outputs.hidden_states
    last_index = len(hidden_states) - 1
    per_layer_predictions = []

    with torch.no_grad():
        for layer_ix, layer_hidden in enumerate(hidden_states):
            last_token_hidden = layer_hidden[:, -1, :]
            if layer_ix == last_index:
                normed = last_token_hidden  # already normed by the model itself
            else:
                normed = model.model.norm(last_token_hidden)
            logits = model.lm_head(normed)
            probs = torch.softmax(logits, dim=-1)[0]
            top_probs, top_ids = torch.topk(probs, top_k)
            layer_preds = [
                (tokenizer.decode([tid.item()]), prob.item())
                for tid, prob in zip(top_ids, top_probs)
            ]
            per_layer_predictions.append(layer_preds)

    return per_layer_predictions


# ============================
# Plotting
# ============================

def plot_degenerate_rate_by_layer(entity_rates, output_path):
    """
    entity_rates: dict {entity_type: [fraction_of_molecules_with_top1=='1'
    per layer]}. One line per entity, so functional_group's rate (if
    supplied) and pka/pkah/mw's rates are directly visible on the same axes.
    """
    plt.figure(figsize=(8, 5))
    sns.set(style='whitegrid')
    colors = sns.color_palette('husl', max(len(entity_rates), 1))
    for i, (entity_type, rates) in enumerate(entity_rates.items()):
        layers = list(range(len(rates)))
        plt.plot(layers, rates, marker='o', markersize=3, linewidth=1.5,
                 label=entity_type, color=colors[i])
    plt.xlabel('Layer index (0 = embedding layer)', fontsize=12)
    plt.ylabel("Fraction of molecules with top-1 token == '1'", fontsize=12)
    plt.title("Degenerate '1'-token bias by layer and entity -- 8B", fontsize=13)
    plt.ylim(0, 1)
    plt.legend()
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

    representatives = pick_fixed_representative_molecules(CSV_PATH, REPRESENTATIVE_CLASS_COLUMN)
    print(f"Using {len(representatives)} fixed representative molecules "
          f"(one per {REPRESENTATIVE_CLASS_COLUMN} class), same set as the "
          f"functional_group logit-lens run.")

    entity_degenerate_rates = {}

    for entity_type in NUMERIC_ENTITIES_TO_TRACE:
        print(f"\n=== Entity: {entity_type} ===")
        template = load_entity_first_template(CONFIG_YAML_PATH, entity_type)

        summary_rows = []
        n_layers_recorded = NUM_LAYERS + 1
        degenerate_by_layer = np.zeros(n_layers_recorded, dtype=int)
        n_molecules = 0

        for class_label, row in representatives.iterrows():
            prompt = template.format(iupac_name=row['iupac_name'], formula=row['formula'])
            print(f"\n  Molecule: {row['iupac_name']} ({row['formula']}) "
                  f"[fg class: {class_label}]")
            print(f"  Prompt: \"{prompt}\"")

            per_layer_predictions = run_logit_lens(model, tokenizer, prompt, device)
            n_molecules += 1

            for layer_ix, top_k_preds in enumerate(per_layer_predictions):
                top1_text, top1_prob = top_k_preds[0]
                top1_clean = top1_text.strip()
                is_degenerate = (top1_clean == '1')
                degenerate_by_layer[layer_ix] += int(is_degenerate)

                summary_rows.append({
                    'entity_type': entity_type,
                    'fg_class': class_label,
                    'molecule': row['iupac_name'],
                    'layer': layer_ix,
                    'top1_token': top1_clean,
                    'top1_prob': round(top1_prob, 4),
                    'top5_tokens': '; '.join(f"{t.strip()}({p:.2f})" for t, p in top_k_preds),
                    'is_degenerate_1': is_degenerate,
                })

                if layer_ix % 8 == 0 or layer_ix == n_layers_recorded - 1:
                    print(f"    Layer {layer_ix:>2}: top1='{top1_clean}' "
                          f"(p={top1_prob:.2f}) {'[DEGENERATE 1]' if is_degenerate else ''}")

        summary_csv_path = os.path.join(OUTPUT_DIR, f'logit_lens_{entity_type}_summary.csv')
        with open(summary_csv_path, 'w', newline='') as f:
            fieldnames = ['entity_type', 'fg_class', 'molecule', 'layer',
                          'top1_token', 'top1_prob', 'top5_tokens', 'is_degenerate_1']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\n  Saved {summary_csv_path}")

        degenerate_rates = (degenerate_by_layer / n_molecules).tolist()
        entity_degenerate_rates[entity_type] = degenerate_rates

        worst_layer = int(np.argmax(degenerate_rates))
        print(f"  Worst '1'-bias layer: {worst_layer}, rate = {degenerate_rates[worst_layer]:.3f} "
              f"({degenerate_by_layer[worst_layer]}/{n_molecules} molecules)")

    plot_degenerate_rate_by_layer(
        entity_degenerate_rates,
        os.path.join(OUTPUT_DIR, 'logit_lens_degenerate_1_rate_pka_pkah_mw.png')
    )
    print(f"\nSaved comparison plot to "
          f"{os.path.join(OUTPUT_DIR, 'logit_lens_degenerate_1_rate_pka_pkah_mw.png')}")

    print("\n=== Summary: average '1'-bias rate across all 33 layers, per entity ===")
    for entity_type, rates in entity_degenerate_rates.items():
        print(f"  {entity_type:<12} avg rate = {np.mean(rates):.3f}, "
              f"peak rate = {np.max(rates):.3f} at layer {int(np.argmax(rates))}")


if __name__ == "__main__":
    main()

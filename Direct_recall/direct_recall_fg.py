"""
Direct recall analysis for the functional-group project (Llama-3.1-8B).

For each of the entity types defined in config_extract_activation_fg_8B.yaml,
this tests whether the model's OWN activations (from prompts asking about
that property) linearly encode that SAME property -- "direct recall",
mirroring Section 5.1 of the "Do Llamas understand the periodic table?"
paper and this project's own categorical_probe.py / basic_linear_regression.py
scripts from the periodic-table work.

- Continuous properties (mw, pka, pkah, tpsa, avg_carbon_oxidation_state,
  hbd, hba, boiling_point_c) -> linear SVR regression, scored by R^2.
- Categorical properties (functional_group, functional_group_structure,
  water_solubility) -> linear SVC classification, scored by accuracy.

Cross-validation uses GroupKFold, grouping rows by molecule (not by
template), so that the same molecule's repeated-template rows never appear
in both the train and test fold of the same split. This matches
categorical_probe_cv() / train_svr_cv() from the periodic-table scripts and
prevents near-duplicate rows from leaking across the split.
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

from sklearn.svm import SVR, SVC
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

# ============================
# User Configuration Variables
# ============================

# Run this script while standing INSIDE the functional_groups/ folder.
CONFIG_YAML_PATH = '../config_extract_activation_fg_8B.yaml'
CSV_PATH = '../functional_group_dataset_2.csv'
ACTIVATIONS_BASE_DIR = '../activation_datasets_functional_groups_2/meta-llama-Meta-Llama-3.1-8B/'
OUTPUT_DIR = '../Results/direct_recall_fg_8B'
SUMMARY_CSV_PATH = os.path.join(OUTPUT_DIR, 'direct_recall_summary.csv')

NUM_LAYERS = 32       # 8B has 32 layers (0-31)
N_SPLITS = 5           # GroupKFold splits, matches categorical_probe.py / basic_linear_regression.py
SVM_C = 2               # matches the C used in the periodic-table scripts


# ============================
# Config loading
# ============================

def load_entity_configs(yaml_path):
    """
    Read entity_type / prompt_name / template count directly from the
    extraction config, so this never drifts out of sync with the actual
    YAML (instead of hardcoding a duplicate table in this script).
    """
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    entities = config['extraction']['entities']
    entity_configs = []
    for entry in entities:
        entity_configs.append({
            'entity_type': entry['entity_type'],
            'prompt_name': entry['prompt_name'],
            'templates_per_molecule': len(entry['templates']),
        })
    return entity_configs


# ============================
# Data loading
# ============================

def load_labels(csv_path, label_column, templates_per_molecule):
    """
    Load the target column and repeat each value templates_per_molecule
    times, to line up with the flattened (molecule, template) activation
    order produced by generate_prompts() in extract_activations.py.

    Numeric columns are coerced with pd.to_numeric(errors='coerce') first,
    since some columns (e.g. pkah) mix numeric values with explanatory text
    like "N/A (no stable aqueous conjugate acid)" for rows where the
    property doesn't apply -- without this coercion, pandas would treat the
    whole column as non-numeric/categorical.
    """
    df = pd.read_csv(csv_path)
    coerced = pd.to_numeric(df[label_column], errors='coerce')
    # A column counts as numeric if it was already numeric dtype, OR if
    # coercing to numeric successfully parsed at least some values (handles
    # mixed columns like pkah, which has real numbers plus explanatory text
    # such as "N/A (no stable aqueous conjugate acid)" for rows where the
    # property doesn't apply).
    is_numeric = pd.api.types.is_numeric_dtype(df[label_column]) or coerced.notna().any()

    if is_numeric:
        # Keep genuinely missing / not-applicable values as NaN rather than
        # filling with a sentinel like -inf: sklearn's SVR.fit() rejects
        # infinite target values outright (ValueError), and treating "does
        # not apply" (e.g. pKaH for alkanes, which have no basic site) as an
        # extreme numeric placeholder isn't chemically meaningful anyway.
        # Rows with NaN targets are dropped from this probe in main().
        labels = coerced.astype(float).values
    else:
        labels = df[label_column].fillna('Unknown').astype(str).values

    labels_repeated = np.repeat(labels, templates_per_molecule)
    return labels_repeated, is_numeric


def build_layer_filename(entity_type, prompt_name, layer_ix):
    """
    Filename format from save_activations(): '{entity_type}.{aggregation}.{prompt_name}.layer_{layer_ix}.pt'
    aggregation is fixed as 'last' per the current config.
    """
    return f'{entity_type}.last.{prompt_name}.layer_{layer_ix}.pt'


def load_activation(activations_dir, entity_type, prompt_name, layer_ix, expected_n):
    """
    Load activations for one layer and convert to numpy array.
    map_location='cpu' is required because these tensors were saved on a
    GPU node during extraction, but this script runs on CPU only.
    """
    filename = build_layer_filename(entity_type, prompt_name, layer_ix)
    file_path = os.path.join(activations_dir, filename)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Activation file not found: {file_path}")

    activation = torch.load(file_path, map_location='cpu').numpy()
    if activation.shape[0] != expected_n:
        raise ValueError(f"Expected {expected_n} rows, got {activation.shape[0]} in {file_path}")
    return activation


# ============================
# Probing (direct recall)
# ============================

def make_groups(n_molecules, templates_per_molecule):
    """
    One group id per molecule, repeated across its templates. Passed to
    GroupKFold so a molecule's repeated-template rows always land together
    on the same side of the train/test split.
    """
    return np.repeat(np.arange(n_molecules), templates_per_molecule)


def regression_probe_cv(X, y, groups, n_splits=N_SPLITS, c=SVM_C):
    """
    Linear SVR with GroupKFold cross-validation. Returns the average R^2
    across folds. Mirrors train_svr_cv() in basic_linear_regression.py.
    """
    gkf = GroupKFold(n_splits=n_splits)
    r2_scores = []
    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])

        sample_weights = compute_sample_weight('balanced', y[train_idx])
        model = SVR(kernel='linear', C=c)
        model.fit(X_train, y[train_idx], sample_weight=sample_weights)
        y_pred = model.predict(X_test)
        r2_scores.append(r2_score(y[test_idx], y_pred))
    return float(np.mean(r2_scores))


def classification_probe_cv(X, y_encoded, groups, n_splits=N_SPLITS, c=SVM_C):
    """
    Linear SVC with GroupKFold cross-validation. Returns the average
    accuracy across folds. Mirrors categorical_probe_cv() in
    categorical_probe.py.
    """
    gkf = GroupKFold(n_splits=n_splits)
    accuracies = []
    for train_idx, test_idx in gkf.split(X, y_encoded, groups=groups):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])

        sample_weights = compute_sample_weight('balanced', y_encoded[train_idx])
        model = SVC(kernel='linear', C=c, class_weight='balanced')
        model.fit(X_train, y_encoded[train_idx], sample_weight=sample_weights)
        y_pred = model.predict(X_test)
        accuracies.append(accuracy_score(y_encoded[test_idx], y_pred))
    return float(np.mean(accuracies))


# ============================
# Plotting
# ============================

def plot_trends(scores_by_entity, output_path, ylabel, title):
    plt.figure(figsize=(8, 5))
    sns.set(style='whitegrid')
    colors = sns.color_palette('husl', max(len(scores_by_entity), 1))
    for i, (entity_type, scores) in enumerate(scores_by_entity.items()):
        layers = list(range(len(scores)))
        plt.plot(layers, scores, marker='o', markersize=3, linewidth=1,
                 label=entity_type, color=colors[i])
    plt.xlabel('Layer Index', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=13)
    plt.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.grid(True, linestyle='--', linewidth=0.7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()


# ============================
# Main
# ============================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    entity_configs = load_entity_configs(CONFIG_YAML_PATH)
    print(f"Loaded {len(entity_configs)} entity configs from {CONFIG_YAML_PATH}")

    df = pd.read_csv(CSV_PATH)
    n_molecules = len(df)

    summary_rows = []
    regression_scores = {}
    classification_scores = {}

    for entity_cfg in entity_configs:
        entity_type = entity_cfg['entity_type']
        prompt_name = entity_cfg['prompt_name']
        templates_per_molecule = entity_cfg['templates_per_molecule']
        activations_dir = os.path.join(ACTIVATIONS_BASE_DIR, entity_type)

        # Direct recall: the target label is the SAME property that was
        # asked about in this entity's own prompts.
        if entity_type not in df.columns:
            print(f"\n=== Entity: {entity_type} -- no matching CSV column, skipping. ===")
            continue

        print(f"\n=== Entity: {entity_type} ({templates_per_molecule} templates/molecule) ===")

        labels_repeated, is_numeric = load_labels(CSV_PATH, entity_type, templates_per_molecule)
        expected_n = n_molecules * templates_per_molecule
        groups = make_groups(n_molecules, templates_per_molecule)

        if is_numeric:
            # Drop rows where the property doesn't apply (NaN target) --
            # e.g. pKaH is undefined for alkanes/alkyl halides. This must
            # happen per-entity since a molecule missing one property may
            # still be valid for every other entity's probe.
            valid_mask = ~np.isnan(labels_repeated)
            n_dropped_molecules = (n_molecules * templates_per_molecule - valid_mask.sum()) // templates_per_molecule
            if n_dropped_molecules > 0:
                print(f"  Dropping {n_dropped_molecules} molecule(s) with no valid '{entity_type}' value.")
            y = labels_repeated
        else:
            le = LabelEncoder()
            y = le.fit_transform(labels_repeated)
            valid_mask = np.ones(len(y), dtype=bool)

        if not os.path.isdir(activations_dir):
            print(f"  Activations dir not found: {activations_dir}, skipping.")
            continue

        layer_scores = []
        for layer_ix in range(NUM_LAYERS):
            try:
                X = load_activation(activations_dir, entity_type, prompt_name, layer_ix, expected_n)
            except (FileNotFoundError, ValueError) as e:
                print(f"  Layer {layer_ix}: {e}")
                layer_scores.append(np.nan)
                continue

            X_valid = X[valid_mask]
            y_valid = y[valid_mask]
            groups_valid = groups[valid_mask]

            if is_numeric:
                score = regression_probe_cv(X_valid, y_valid, groups_valid)
            else:
                score = classification_probe_cv(X_valid, y_valid, groups_valid)
            layer_scores.append(score)
            metric_name = 'R2' if is_numeric else 'accuracy'
            print(f"  Layer {layer_ix}: {metric_name} = {score:.3f}")

        if all(np.isnan(s) for s in layer_scores):
            print(f"  No valid layers for {entity_type}, skipping summary/plot.")
            continue

        best_layer = int(np.nanargmax(layer_scores))
        best_score = layer_scores[best_layer]
        print(f"  Best layer: {best_layer}, score = {best_score:.3f}")

        for layer_ix, score in enumerate(layer_scores):
            summary_rows.append({
                'entity_type': entity_type,
                'target_type': 'regression' if is_numeric else 'classification',
                'layer': layer_ix,
                'score': score,
            })

        if is_numeric:
            regression_scores[entity_type] = layer_scores
        else:
            classification_scores[entity_type] = layer_scores

    if not summary_rows:
        print("\nNo results to summarize.")
        return

    with open(SUMMARY_CSV_PATH, 'w', newline='') as f:
        fieldnames = ['entity_type', 'target_type', 'layer', 'score']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nSaved summary table to {SUMMARY_CSV_PATH}")

    if regression_scores:
        plot_trends(regression_scores, os.path.join(OUTPUT_DIR, 'direct_recall_regression_r2.png'),
                    ylabel='R^2', title='Direct Recall (Regression) -- 8B')
    if classification_scores:
        plot_trends(classification_scores, os.path.join(OUTPUT_DIR, 'direct_recall_classification_accuracy.png'),
                    ylabel='Accuracy', title='Direct Recall (Classification) -- 8B')

    print("\n=== Best layer per entity (sorted by score, descending) ===")
    best_rows = []
    for entity_type, scores in {**regression_scores, **classification_scores}.items():
        best_layer = int(np.nanargmax(scores))
        best_rows.append((entity_type, best_layer, scores[best_layer]))
    for entity_type, layer, score in sorted(best_rows, key=lambda r: -r[2]):
        print(f"  {entity_type:<28} best layer {layer:>2}  score={score:.3f}")


if __name__ == "__main__":
    main()

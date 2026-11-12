"""
Direct recall analysis for the functional-group project (Llama-3.1-8B).


Evaluation uses 5-fold functional-group-held-out splits. For each fold,
multiple functional groups are held out together. One molecule from each
eligible held-out functional group is added back to training as a support
example, while the remaining molecules from those groups form the test set.

Support molecules are randomly selected across multiple repeats so the
result is not determined by one arbitrary support choice.


- Continuous properties (mw, pka, pkah, tpsa, avg_carbon_oxidation_state,
  hbd, hba, boiling_point_c) -> linear SVR regression, scored by R^2.
- Categorical properties (functional_group, functional_group_structure,
  water_solubility) -> linear SVC classification, scored by accuracy.



Unlike strict functional-group-held-out classification, this one-shot setup
can evaluate the functional_group target because every tested class has one
support molecule in training.
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

from sklearn.svm import SVR, SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, accuracy_score, mean_absolute_error
from sklearn.utils.class_weight import compute_sample_weight

# ============================
# User Configuration Variables
# ============================

# Run this script while standing inside Functional_groups/Direct_recall.
CONFIG_YAML_PATH = '../config_extract_activation_fg_8B.yaml'
CSV_PATH = '../functional_group_dataset_2.csv'
ACTIVATIONS_BASE_DIR = '../activation_datasets_functional_groups_2/meta-llama-Llama-3.1-8B/'
OUTPUT_DIR = '../Results/direct_recall_fg_multigroup_one_shot_8B'
SUMMARY_CSV_PATH = os.path.join(
    OUTPUT_DIR,
    'direct_recall_fg_multigroup_one_shot_summary.csv',
)

NUM_LAYERS = 32       # 8B has 32 layers (0-31)
N_SPLITS = 5
SVM_C = 2               # matches the C used in the periodic-table scripts
HOLDOUT_GROUP_COLUMN = 'functional_group'
N_SUPPORT_REPEATS = 20
SUPPORT_RANDOM_SEED = 42

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

def make_functional_group_metadata(
    df,
    templates_per_molecule,
    group_column=HOLDOUT_GROUP_COLUMN,
):
    """
    Return functional-group labels and molecule IDs repeated across
    all prompt templates.
    """
    if group_column not in df.columns:
        raise KeyError(
            f"Required holdout column '{group_column}' was not found "
            "in the dataset."
        )

    if df[group_column].isna().any():
        missing_count = int(df[group_column].isna().sum())

        raise ValueError(
            f"The holdout column '{group_column}' contains "
            f"{missing_count} missing value(s)."
        )

    functional_groups = df[group_column].astype(str).values
    molecule_ids = np.arange(len(df), dtype=int)

    return (
        np.repeat(
            functional_groups,
            templates_per_molecule,
        ),
        np.repeat(
            molecule_ids,
            templates_per_molecule,
        ),
    )

def build_one_shot_splits(
    functional_groups,
    molecule_ids,
    n_splits=N_SPLITS,
    n_support_repeats=N_SUPPORT_REPEATS,
    base_seed=SUPPORT_RANDOM_SEED,
):
    """
    Match the strict FG-held-out 5-fold design, but add one support
    molecule from EACH held-out functional group back into training.

    Example fold:
        held-out groups = alcohol, amine, acid, ether

    For each repeat:
        training =
            all molecules outside those held-out groups
            + one alcohol
            + one amine
            + one acid
            + one ether

        testing =
            all remaining molecules from those held-out groups

    Support molecules are randomly selected multiple times so the result
    is not determined by one arbitrary choice.
    """

    splits = []

    gkf = GroupKFold(
        n_splits=min(
            n_splits,
            len(np.unique(functional_groups)),
        )
    )

    dummy_X = np.zeros(len(functional_groups))

    for fold_ix, (base_train_idx, heldout_idx) in enumerate(
        gkf.split(
            dummy_X,
            groups=functional_groups,
        )
    ):

        heldout_groups = np.unique(
            functional_groups[heldout_idx]
        )

        for repeat_ix in range(n_support_repeats):

            rng = np.random.default_rng(
                base_seed
                + fold_ix * 1000
                + repeat_ix
            )

            support_molecules = []
            supported_groups = []

            for heldout_group in heldout_groups:

                group_rows = heldout_idx[
                    functional_groups[heldout_idx]
                    == heldout_group
                ]

                group_molecules = np.unique(
                    molecule_ids[group_rows]
                )

                # Need at least 2 molecules:
                # 1 support + at least 1 test molecule
                if len(group_molecules) < 2:
                    continue

                support_molecule = int(
                    rng.choice(group_molecules)
                )

                support_molecules.append(
                    support_molecule
                )

                supported_groups.append(
                    heldout_group
                )
            if not support_molecules:
                continue

            support_molecules = np.asarray(
                support_molecules,
                dtype=int,
            )

            # Include every template belonging to each support molecule.
            support_mask = np.isin(
                molecule_ids,
                support_molecules,
            )

            support_idx = np.flatnonzero(
                support_mask
            )

            # One-shot training:
            # normal zero-shot training + support molecules.
            train_idx = np.unique(
                np.concatenate([
                    base_train_idx,
                    support_idx,
                ])
            )

            # Test uses the SAME held-out fold, except support molecules
            # themselves are removed.
            heldout_mask = np.zeros(
                len(functional_groups),
                dtype=bool,
            )

            heldout_mask[heldout_idx] = True

            supported_group_mask = np.isin(
                functional_groups,
                supported_groups,
            )

            test_mask = (
                heldout_mask
                & supported_group_mask
                & ~support_mask
            )

            test_idx = np.flatnonzero(
                test_mask
            )

            if len(train_idx) and len(test_idx):
                splits.append(
                    (
                        train_idx,
                        test_idx,
                        fold_ix,
                        tuple(support_molecules.tolist()),
                    )
                )

    return splits


def regression_probe_one_shot(
    X,
    y,
    functional_groups,
    molecule_ids,
    c=SVM_C,
):
    """
    Linear SVR evaluated across all rotated one-support splits.

    MAE is calculated for every valid one-shot split.

    R^2 is calculated only when the test set contains at least
    two distinct target values.
    """
    r2_scores = []
    mae_scores = []

    n_r2_skipped = 0

    for train_idx, test_idx, _, _ in build_one_shot_splits(
        functional_groups,
        molecule_ids,
    ):
        scaler = StandardScaler()

        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])

        sample_weights = compute_sample_weight(
            'balanced',
            y[train_idx],
        )

        model = SVR(kernel='linear', C=c)

        model.fit(
            X_train,
            y[train_idx],
            sample_weight=sample_weights,
        )

        y_pred = model.predict(X_test)

        # MAE is valid even if every test target is identical.
        mae_scores.append(
            float(
                mean_absolute_error(
                    y[test_idx],
                    y_pred,
                )
            )
        )

        # R^2 requires target variation.
        if len(np.unique(y[test_idx])) >= 2:
            score = float(
                r2_score(
                    y[test_idx],
                    y_pred,
                )
            )

            if np.isfinite(score):
                r2_scores.append(score)
            else:
                n_r2_skipped += 1

        else:
            n_r2_skipped += 1

    if not mae_scores:
        raise ValueError(
            "No valid one-shot regression splits remained."
        )

    mean_r2 = (
        float(np.mean(r2_scores))
        if r2_scores
        else np.nan
    )

    std_r2 = (
        float(np.std(r2_scores))
        if r2_scores
        else np.nan
    )

    return (
        mean_r2,
        std_r2,
        float(np.mean(mae_scores)),
        float(np.std(mae_scores)),
        len(mae_scores),
        len(r2_scores),
        n_r2_skipped,
    )

def classification_probe_one_shot(
    X,
    y_encoded,
    functional_groups,
    molecule_ids,
    c=SVM_C,
):
    """
    Linear SVC evaluated across all rotated one-support splits.
    """
    accuracies = []
    skipped = 0
    for train_idx, test_idx, _, _ in build_one_shot_splits(
        functional_groups, molecule_ids
    ):
        if len(np.unique(y_encoded[train_idx])) < 2:
            skipped += 1
            continue
        unseen_classes = set(y_encoded[test_idx]) - set(y_encoded[train_idx])
        if unseen_classes:
            skipped += 1
            continue
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])

        sample_weights = compute_sample_weight('balanced', y_encoded[train_idx])
        model = SVC(kernel='linear', C=c, class_weight='balanced')
        model.fit(X_train, y_encoded[train_idx], sample_weight=sample_weights)
        y_pred = model.predict(X_test)
        accuracies.append(accuracy_score(y_encoded[test_idx], y_pred))

    if not accuracies:
        raise ValueError("No valid one-shot classification splits remained.")
    return (
        float(np.mean(accuracies)),
        float(np.std(accuracies)),
        len(accuracies),
        skipped,
    )


# ============================
# Plotting
# ============================

def plot_all_entity_panels(
    regression_scores,
    classification_scores,
    output_path,
):
    """
    Plot every one-support direct-recall property in its own panel.

    Regression targets use R^2.
    Classification targets use accuracy.

    Each panel has its own y-axis scale.
    """

    all_entities = []

    # Regression properties
    for entity_type, scores in regression_scores.items():
        all_entities.append(
            (
                entity_type,
                scores,
                'R^2',
            )
        )

    # Classification properties
    for entity_type, scores in classification_scores.items():
        all_entities.append(
            (
                entity_type,
                scores,
                'Accuracy',
            )
        )

    n_entities = len(all_entities)

    if n_entities == 0:
        return

    n_cols = 3
    n_rows = int(
        np.ceil(
            n_entities / n_cols
        )
    )

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(15, 4 * n_rows),
        squeeze=False,
    )

    axes = axes.flatten()

    for ax, (
        entity_type,
        scores,
        metric_name,
    ) in zip(
        axes,
        all_entities,
    ):

        scores = np.asarray(
            scores,
            dtype=float,
        )

        layers = np.arange(
            len(scores)
        )

        ax.plot(
            layers,
            scores,
            marker='o',
            markersize=3,
            linewidth=1,
        )

        # R^2 = 0 reference line
        if metric_name == 'R^2':
            ax.axhline(
                0,
                color='black',
                linewidth=0.8,
            )

        # Mark best layer
        if not np.all(
            np.isnan(scores)
        ):

            best_layer = int(
                np.nanargmax(scores)
            )

            best_score = scores[
                best_layer
            ]

            ax.scatter(
                best_layer,
                best_score,
                marker='*',
                s=150,
                zorder=5,
            )

            ax.set_title(
                f"{entity_type} "
                f"(best {best_score:.2f} "
                f"at layer {best_layer})",
                fontsize=10,
            )

        else:
            ax.set_title(
                entity_type,
                fontsize=10,
            )

        ax.set_xlabel(
            'Layer'
        )

        ax.set_ylabel(
            metric_name
        )

        ax.grid(
            True,
            linestyle='--',
            linewidth=0.6,
            alpha=0.6,
        )

    # Hide unused panels
    for ax in axes[n_entities:]:
        ax.axis('off')

    fig.suptitle(
        'Direct Recall: One Support Molecule per Held-out Functional Group -- Llama-3.1-8B',
        fontsize=15,
    )

    plt.tight_layout(
        rect=[0, 0, 1, 0.96]
    )

    plt.savefig(
        output_path,
        dpi=220,
        bbox_inches='tight',
    )

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

    if HOLDOUT_GROUP_COLUMN not in df.columns:
        raise KeyError(
            f"Required holdout column '{HOLDOUT_GROUP_COLUMN}' was not found "
            f"in {CSV_PATH}."
        )

    n_total_functional_groups = df[HOLDOUT_GROUP_COLUMN].nunique(dropna=True)
    print(
        f"One-molecule functional-group support evaluation using "
        f"{n_total_functional_groups} unique functional groups."
    )

    summary_rows = []
    regression_scores = {}
    regression_stds = {}
    classification_scores = {}
    classification_stds = {}

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
        groups, molecule_ids = make_functional_group_metadata(
            df,
            templates_per_molecule,
        )

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

        groups_valid = groups[valid_mask]
        molecule_ids_valid = molecule_ids[valid_mask]
        y_valid = y[valid_mask]
        n_valid_groups = len(np.unique(groups_valid))

        if n_valid_groups < 2:
            print(
                f"  Only {n_valid_groups} valid functional group(s) remain "
                f"for '{entity_type}'; at least 2 are required. Skipping."
            )
            continue

        one_shot_splits = build_one_shot_splits(
            groups_valid,
            molecule_ids_valid,
        )
        if not one_shot_splits:
            print(
                "  No functional group has at least two valid molecules; "
                "one-shot evaluation is impossible. Skipping."
            )
            continue
        print(
            f"  Five-fold FG one-support evaluation: "
            f"{n_valid_groups} valid groups, "
            f"{len(one_shot_splits)} fold/support evaluations."
        )

        if not os.path.isdir(activations_dir):
            print(f"  Activations dir not found: {activations_dir}, skipping.")
            continue

        layer_scores = []
        layer_stds = []

        layer_maes = []
        layer_mae_stds = []

        layer_mae_split_counts = []
        layer_r2_split_counts = []
        layer_r2_skipped_counts = []
        for layer_ix in range(NUM_LAYERS):
            try:
                X = load_activation(activations_dir, entity_type, prompt_name, layer_ix, expected_n)
            except (FileNotFoundError, ValueError) as e:
                print(f"  Layer {layer_ix}: {e}")
                layer_scores.append(np.nan)
                layer_stds.append(np.nan)

                layer_maes.append(np.nan)
                layer_mae_stds.append(np.nan)

                layer_mae_split_counts.append(0)
                layer_r2_split_counts.append(0)
                layer_r2_skipped_counts.append(0)

                continue

            X_valid = X[valid_mask]

            try:
                if is_numeric:
                    (
                        score,
                        score_std,
                        mae,
                        mae_std,
                        n_mae_splits,
                        n_r2_splits,
                        n_r2_skipped,
                    ) = regression_probe_one_shot(
                        X_valid,
                        y_valid,
                        groups_valid,
                        molecule_ids_valid,
                    )
                else:
                    score, score_std, n_evaluation_splits, n_skipped_splits = classification_probe_one_shot(
                        X_valid,
                        y_valid,
                        groups_valid,
                        molecule_ids_valid,
                    )
                    mae = np.nan
                    mae_std = np.nan

                    n_mae_splits = np.nan
                    n_r2_splits = np.nan
                    n_r2_skipped = np.nan
            except ValueError as e:
                print(f"  Layer {layer_ix}: {e}")
                layer_scores.append(np.nan)
                layer_stds.append(np.nan)
                layer_maes.append(np.nan)
                layer_mae_stds.append(np.nan)
                layer_mae_split_counts.append(0)
                layer_r2_split_counts.append(0)
                layer_r2_skipped_counts.append(0)
                continue
            layer_scores.append(score)
            layer_stds.append(score_std)

            layer_maes.append(mae)
            layer_mae_stds.append(mae_std)

            layer_mae_split_counts.append(n_mae_splits)
            layer_r2_split_counts.append(n_r2_splits)
            layer_r2_skipped_counts.append(n_r2_skipped)
            if is_numeric:
                print(
                    f"  Layer {layer_ix}: "
                    f"R2 = {score:.3f}, "
                    f"MAE = {mae:.3f}, "
                    f"R2 SD = {score_std:.3f}, "
                    f"MAE SD = {mae_std:.3f}, "
                    f"MAE splits = {n_mae_splits}, "
                    f"R2 splits = {n_r2_splits}, "
                    f"R2 skipped = {n_r2_skipped}"
                )
            else:
                print(
                    f"  Layer {layer_ix}: "
                    f"accuracy = {score:.3f}"
                )

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
                'holdout_group_column': HOLDOUT_GROUP_COLUMN,
                'n_holdout_groups': n_valid_groups,
                'split_strategy': 'five_fold_one_support_per_held_out_functional_group',
                'support_molecules_per_group': 1,

                'layer': layer_ix,

                'score': score,
                'r2': score if is_numeric else np.nan,
                'score_std': layer_stds[layer_ix],

                'mae': layer_maes[layer_ix],
                'mae_std': layer_mae_stds[layer_ix],

                'n_mae_splits': layer_mae_split_counts[layer_ix],
                'n_r2_splits': layer_r2_split_counts[layer_ix],
                'n_r2_skipped': layer_r2_skipped_counts[layer_ix],
            })
        if is_numeric:
            regression_scores[entity_type] = layer_scores
            regression_stds[entity_type] = layer_stds
        else:
            classification_scores[entity_type] = layer_scores
            classification_stds[entity_type] = layer_stds

    if not summary_rows:
        print("\nNo results to summarize.")
        return

    with open(SUMMARY_CSV_PATH, 'w', newline='') as f:
        fieldnames = [
            'entity_type',
            'target_type',
            'holdout_group_column',
            'n_holdout_groups',
            'split_strategy',
            'support_molecules_per_group',

            'layer',

            'score',
            'r2',
            'score_std',

            'mae',
            'mae_std',

            'n_mae_splits',
            'n_r2_splits',
            'n_r2_skipped',
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nSaved summary table to {SUMMARY_CSV_PATH}")

    plot_all_entity_panels(
        regression_scores,
        classification_scores,
        os.path.join(
            OUTPUT_DIR,
            'direct_recall_fg_multigroup_one_shot_all_panels.png',
        ),
    )

    print("\n=== Best layer per entity (sorted by score, descending) ===")
    best_rows = []
    for entity_type, scores in {**regression_scores, **classification_scores}.items():
        best_layer = int(np.nanargmax(scores))
        best_rows.append((entity_type, best_layer, scores[best_layer]))
    for entity_type, layer, score in sorted(best_rows, key=lambda r: -r[2]):
        print(f"  {entity_type:<28} best layer {layer:>2}  score={score:.3f}")


if __name__ == "__main__":
    main()

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
ACTIVATIONS_BASE_DIR = '../activation_datasets_functional_groups_2/meta-llama-Llama-3.1-8B/'
OUTPUT_DIR = '../Results/direct_recall_fg_8B_test_random_null'
SUMMARY_CSV_PATH = os.path.join(OUTPUT_DIR, 'direct_recall_summary.csv')

NUM_LAYERS = 32       # 8B has 32 layers (0-31)
N_SPLITS = 5           # GroupKFold splits, matches categorical_probe.py / basic_linear_regression.py
SVM_C = 2               # matches the C used in the periodic-table scripts

# ============================
# Null-baseline settings
# ============================

RUN_NULL_BASELINE = True

# Use "random" or "permute_molecules" to create the null baseline. 
# "random" generates completely random Gaussian vectors, while "permute_molecules" randomly reassigns complete molecule activation blocks  
NULL_MODE = "random"

# Seeds 0 through 19 give 20 different molecule permutations.
NULL_SEEDS = list(range(20)) # Use a smaller number of seeds for testing; increase to 20 for full runs.

# Used for the shaded null interval in the result plots.
NULL_PERCENTILES = (5, 95)

# Additional result files.
NULL_RUNS_CSV_PATH = os.path.join(
    OUTPUT_DIR,
    f"direct_recall_{NULL_MODE}_runs.csv",
)

SUMMARY_WITH_NULL_CSV_PATH = os.path.join(
    OUTPUT_DIR,
    f"direct_recall_summary_with_{NULL_MODE}.csv",
)

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

def permute_molecule_activations(
    X,
    n_molecules,
    templates_per_molecule,
    seed,
):
    """
    Randomly reassign complete molecule activation blocks.

    All templates belonging to a molecule remain together, but the complete
    block is assigned to a different molecule position. Using the same seed
    at every layer produces the same molecule reassignment across layers.
    """
    expected_rows = n_molecules * templates_per_molecule

    if X.shape[0] != expected_rows:
        raise ValueError(
            f"Cannot permute activation blocks: expected {expected_rows} "
            f"rows from {n_molecules} molecules x "
            f"{templates_per_molecule} templates, got {X.shape[0]}."
        )

    # Convert from:
    # [activation rows, hidden dimension]
    #
    # to:
    # [molecules, templates, hidden dimension]
    molecule_blocks = X.reshape(
        n_molecules,
        templates_per_molecule,
        X.shape[1],
    )

    rng = np.random.default_rng(seed)
    permutation = rng.permutation(n_molecules)

    permuted_blocks = molecule_blocks[permutation]

    # Return to the format required by the probe.
    return permuted_blocks.reshape(X.shape)

def create_random_activations(
    X,
    seed,
    layer_ix,
):
    """
    Create completely random Gaussian vectors with the same shape as the
    real activation matrix.

    The random vectors contain no molecule, template, layer, or chemical
    information. SeedSequence makes each seed/layer combination reproducible
    while generating different random vectors for different layers.
    """
    seed_sequence = np.random.SeedSequence(
        [int(seed), int(layer_ix)]
    )
    rng = np.random.default_rng(seed_sequence)

    return rng.standard_normal(X.shape).astype(np.float32)

def create_null_activations(
    X,
    n_molecules,
    templates_per_molecule,
    seed,
    layer_ix,
    mode,
):
    """
    Create the selected null activation representation.
    """
    if mode == "permute_molecules":
        return permute_molecule_activations(
            X=X,
            n_molecules=n_molecules,
            templates_per_molecule=templates_per_molecule,
            seed=seed,
        )

    if mode == "random":
        return create_random_activations(
            X=X,
            seed=seed,
            layer_ix=layer_ix,
        )

    raise ValueError(
        f"Unknown NULL_MODE '{mode}'. Use either "
        "'permute_molecules' or 'random'."
    )

def summarize_null_scores(null_scores):
    """
    Summarize the probe scores produced by all random seeds.
    """
    scores = np.asarray(null_scores, dtype=float)

    if scores.size == 0:
        raise ValueError("No null scores were provided.")

    low_percentile, high_percentile = NULL_PERCENTILES

    return {
        "null_mean": float(np.mean(scores)),
        "null_seed_sd": (
            float(np.std(scores, ddof=1))
            if len(scores) > 1
            else 0.0
        ),
        "null_p05": float(np.percentile(scores, low_percentile)),
        "null_p95": float(np.percentile(scores, high_percentile)),
    }


def empirical_p_value(real_score, null_scores):
    """
    Fraction of null runs that performed at least as well as the real data.

    The +1 correction prevents a p-value of exactly zero when none of the
    tested null permutations exceeds the real score.
    """
    scores = np.asarray(null_scores, dtype=float)

    return float(
        (1 + np.sum(scores >= real_score))
        / (len(scores) + 1)
    )

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

def plot_real_vs_null(
    entity_type,
    plot_data,
    output_dir,
    null_mode,
    number_of_seeds,
):
    """
    Plot the real activation score against the multi-seed null baseline.
    """
    real_scores = np.asarray(
        plot_data["real_scores"],
        dtype=float,
    )
    null_means = np.asarray(
        plot_data["null_means"],
        dtype=float,
    )
    null_p05 = np.asarray(
        plot_data["null_p05"],
        dtype=float,
    )
    null_p95 = np.asarray(
        plot_data["null_p95"],
        dtype=float,
    )

    layers = np.arange(len(real_scores))

    figure, axis = plt.subplots(figsize=(8, 5))

    # Real activation result
    axis.plot(
        layers,
        real_scores,
        color="#1f77b4",
        marker="o",
        markersize=3,
        linewidth=2,
        label="Real activations",
    )

    # Mean null result across seeds
    axis.plot(
        layers,
        null_means,
        color="#ff7f0e",
        linestyle="--",
        linewidth=2,
        label=(
            f"{null_mode.replace('_', ' ')} mean "
            f"({number_of_seeds} seeds)"
        ),
    )

    # Null percentile interval
    axis.fill_between(
        layers,
        null_p05,
        null_p95,
        color="#ff7f0e",
        alpha=0.20,
        label=(
            f"Null {NULL_PERCENTILES[0]}th–"
            f"{NULL_PERCENTILES[1]}th percentile"
        ),
    )

    target_type = plot_data["target_type"]

    if target_type == "classification":
        axis.set_ylabel("Grouped-CV accuracy")
        axis.set_ylim(0.0, 1.0)
    else:
        axis.axhline(
            0.0,
            color="gray",
            linestyle=":",
            linewidth=1,
            label="$R^2 = 0$",
        )
        axis.set_ylabel("Grouped-CV $R^2$")

    axis.set_xlabel("Layer index")
    axis.set_title(
        f"Direct recall: {entity_type}\n"
        f"Real activations vs. {null_mode.replace('_', ' ')} baseline"
    )

    axis.grid(
        True,
        linestyle="--",
        linewidth=0.6,
        alpha=0.6,
    )
    axis.legend(loc="best", fontsize=8)

    figure.tight_layout()

    safe_entity_name = (
        entity_type
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
    )

    output_path = os.path.join(
        output_dir,
        f"direct_recall_{safe_entity_name}_{null_mode}.png",
    )

    figure.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(figure)

    return output_path

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

    # Individual result from every null seed.
    null_run_rows = []

    # Real score and summarized null results for every layer.
    summary_with_null_rows = []

    # Data that will later be used for the real-versus-null plots.
    null_plot_data = {}

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

        # Store the summarized null values for plotting later.
        layer_null_means = []
        layer_null_seed_sds = []
        layer_null_p05 = []
        layer_null_p95 = []

        for layer_ix in range(NUM_LAYERS):
            try:
                X = load_activation(
                    activations_dir,
                    entity_type,
                    prompt_name,
                    layer_ix,
                    expected_n,
                )
            except (FileNotFoundError, ValueError) as e:
                print(f"  Layer {layer_ix}: {e}")

                layer_scores.append(np.nan)
                layer_null_means.append(np.nan)
                layer_null_seed_sds.append(np.nan)
                layer_null_p05.append(np.nan)
                layer_null_p95.append(np.nan)
                continue

            # Remove molecules without valid target values.
            X_valid = X[valid_mask]
            y_valid = y[valid_mask]
            groups_valid = groups[valid_mask]

            n_valid_molecules = len(np.unique(groups_valid))

            # ---------------------------------
            # 1. Calculate the real-data score
            # ---------------------------------

            if is_numeric:
                real_score = regression_probe_cv(
                    X_valid,
                    y_valid,
                    groups_valid,
                )
                metric_name = "R2"
                target_type = "regression"
            else:
                real_score = classification_probe_cv(
                    X_valid,
                    y_valid,
                    groups_valid,
                )
                metric_name = "accuracy"
                target_type = "classification"

            layer_scores.append(real_score)

            # ---------------------------------
            # 2. Calculate all null-seed scores
            # ---------------------------------

            null_scores = []

            if RUN_NULL_BASELINE:
                for seed in NULL_SEEDS:
                    X_null = create_null_activations(
                        X=X_valid,
                        n_molecules=n_valid_molecules,
                        templates_per_molecule=templates_per_molecule,
                        seed=seed,
                        layer_ix=layer_ix,
                        mode=NULL_MODE,
                    )

                    if is_numeric:
                        null_score = regression_probe_cv(
                            X_null,
                            y_valid,
                            groups_valid,
                        )
                    else:
                        null_score = classification_probe_cv(
                            X_null,
                            y_valid,
                            groups_valid,
                        )

                    null_scores.append(null_score)

                    # Save the individual seed result.
                    null_run_rows.append({
                        "entity_type": entity_type,
                        "target_type": target_type,
                        "layer": layer_ix,
                        "null_mode": NULL_MODE,
                        "seed": seed,
                        "null_score": null_score,
                    })

            # ---------------------------------
            # 3. Summarize all null seeds
            # ---------------------------------

            if null_scores:
                null_summary = summarize_null_scores(null_scores)

                null_mean = null_summary["null_mean"]
                null_seed_sd = null_summary["null_seed_sd"]
                null_p05 = null_summary["null_p05"]
                null_p95 = null_summary["null_p95"]

                p_value = empirical_p_value(
                    real_score,
                    null_scores,
                )
            else:
                null_mean = np.nan
                null_seed_sd = np.nan
                null_p05 = np.nan
                null_p95 = np.nan
                p_value = np.nan

            layer_null_means.append(null_mean)
            layer_null_seed_sds.append(null_seed_sd)
            layer_null_p05.append(null_p05)
            layer_null_p95.append(null_p95)

            # Save the real result together with the null summary.
            summary_with_null_rows.append({
                "entity_type": entity_type,
                "target_type": target_type,
                "layer": layer_ix,
                "real_score": real_score,
                "null_mode": NULL_MODE,
                "n_null_seeds": len(null_scores),
                "null_mean": null_mean,
                "null_seed_sd": null_seed_sd,
                "null_p05": null_p05,
                "null_p95": null_p95,
                "real_minus_null": real_score - null_mean,
                "empirical_p": p_value,
            })

            print(
                f"  Layer {layer_ix}: "
                f"real {metric_name}={real_score:.3f}, "
                f"{NULL_MODE} mean={null_mean:.3f}, "
                f"null SD={null_seed_sd:.3f}, "
                f"p={p_value:.4f}"
            )

        if all(np.isnan(s) for s in layer_scores):
            print(
                f"  No valid layers for {entity_type}, "
                "skipping summary/plot."
            )
            continue

        null_plot_data[entity_type] = {
            "target_type": (
                "regression"
                if is_numeric
                else "classification"
            ),
            "real_scores": layer_scores,
            "null_means": layer_null_means,
            "null_seed_sds": layer_null_seed_sds,
            "null_p05": layer_null_p05,
            "null_p95": layer_null_p95,
        }

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
    # ============================
    # Save individual null runs
    # ============================

    if RUN_NULL_BASELINE and null_run_rows:
        null_run_fieldnames = [
            "entity_type",
            "target_type",
            "layer",
            "null_mode",
            "seed",
            "null_score",
        ]

        with open(
            NULL_RUNS_CSV_PATH,
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=null_run_fieldnames,
            )
            writer.writeheader()
            writer.writerows(null_run_rows)

        print(
            f"Saved individual null runs to "
            f"{NULL_RUNS_CSV_PATH}"
        )
    # ============================
    # Save summarized null results
    # ============================

    if RUN_NULL_BASELINE and summary_with_null_rows:
        summary_with_null_fieldnames = [
            "entity_type",
            "target_type",
            "layer",
            "real_score",
            "null_mode",
            "n_null_seeds",
            "null_mean",
            "null_seed_sd",
            "null_p05",
            "null_p95",
            "real_minus_null",
            "empirical_p",
        ]

        with open(
            SUMMARY_WITH_NULL_CSV_PATH,
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=summary_with_null_fieldnames,
            )
            writer.writeheader()
            writer.writerows(summary_with_null_rows)

        print(
            f"Saved real-versus-null summary to "
            f"{SUMMARY_WITH_NULL_CSV_PATH}"
        )
    # ============================
    # Plot real vs. null results
    # ============================

    if RUN_NULL_BASELINE:
        for entity_type, plot_data in null_plot_data.items():
            plot_path = plot_real_vs_null(
                entity_type=entity_type,
                plot_data=plot_data,
                output_dir=OUTPUT_DIR,
                null_mode=NULL_MODE,
                number_of_seeds=len(NULL_SEEDS),
            )

            print(f"Saved null-comparison plot to {plot_path}")

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

#!/usr/bin/env python3
"""Probe functional-group position across prompt-cue ablation conditions.

The activation extractor stores ten consecutive prompt-template activations
for each molecule. Cross-validation keeps all templates from a molecule in the
same fold. By default, it also keeps the matched position-1/position-2 isomer
pair in the same fold, preventing identical molecular-formula controls from
appearing on opposite sides of a split.

The PCA and scaler are fit inside each training fold to avoid leakage.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:  # Compatibility with older scikit-learn installations.
    StratifiedGroupKFold = None


PROMPT_NAME = "10_templates"
DEFAULT_TEMPLATES_PER_MOLECULE = 10
RANDOM_STATE = 42
BASELINE = 0.5

CONDITIONS: Dict[str, str] = {
    "Original (IUPAC + original notation)": "fg_original",
    "IUPAC only": "fg_iupac_only",
    "Standardized SMILES only": "fg_smiles_only",
    "Molecular formula only": "fg_formula_only",
}


def default_project_root() -> Path:
    return Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    root = default_project_root()
    parser = argparse.ArgumentParser(
        description="Run a grouped linear probe for functional-group position.",
    )
    parser.add_argument(
        "--csv",
        default=str(root / "functional_group_position_ablation_24mol.csv"),
        help="Balanced 24-molecule ablation dataset.",
    )
    parser.add_argument(
        "--activations-root",
        default=str(
            root
            / "activation_datasets_fg_position_ablation"
            / "meta-llama-Meta-Llama-3.1-8B"
        ),
        help="Model directory containing one subdirectory per condition.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(root / "Results" / "fg_position_ablation_probe_8B"),
        help="Directory for CSV and PNG results.",
    )
    parser.add_argument(
        "--templates-per-molecule",
        type=int,
        default=DEFAULT_TEMPLATES_PER_MOLECULE,
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=4,
        help="Number of grouped cross-validation folds (default: 4).",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=20,
        help="Maximum PCA components fit separately inside each fold.",
    )
    parser.add_argument(
        "--group-by",
        choices=("pair", "molecule"),
        default="pair",
        help=(
            "Use matched-pair groups (recommended) or molecule groups. Pair "
            "grouping is stricter and still keeps every molecule intact."
        ),
    )
    parser.add_argument(
        "--layers",
        nargs="*",
        type=int,
        default=None,
        help="Optional layer list. Omit to use every shared available layer.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the dataset and activation files without fitting probes.",
    )
    return parser.parse_args()


def load_dataset(csv_path: Path) -> pd.DataFrame:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    dataframe = pd.read_csv(csv_path)
    required = {
        "iupac_name",
        "functional_group",
        "functional_group_position",
        "carbon_count",
        "pair_id",
    }
    missing = sorted(required - set(dataframe.columns))
    if missing:
        raise ValueError(f"Dataset is missing: {', '.join(missing)}")
    if dataframe["iupac_name"].isna().any() or dataframe["iupac_name"].duplicated().any():
        raise ValueError("iupac_name must be unique and nonmissing")

    dataframe = dataframe.copy()
    dataframe["functional_group_position"] = pd.to_numeric(
        dataframe["functional_group_position"], errors="raise"
    ).astype(int)
    if set(dataframe["functional_group_position"]) != {1, 2}:
        raise ValueError("The target must contain exactly positions 1 and 2")

    position_counts = dataframe["functional_group_position"].value_counts().to_dict()
    if position_counts != {1: 12, 2: 12}:
        raise ValueError(f"Expected 12 molecules per position, found {position_counts}")

    pair_check = dataframe.groupby("pair_id")["functional_group_position"].agg(list)
    invalid_pairs = [
        pair_id for pair_id, values in pair_check.items() if sorted(values) != [1, 2]
    ]
    if invalid_pairs:
        raise ValueError(
            "Every pair_id must contain one position-1 and one position-2 molecule: "
            + ", ".join(invalid_pairs)
        )
    return dataframe


def activation_pattern(entity_type: str) -> re.Pattern[str]:
    return re.compile(
        rf"{re.escape(entity_type)}\.last\.{re.escape(PROMPT_NAME)}\.layer_(\d+)\.pt"
    )


def discover_condition_files(
    activations_root: Path,
    entity_type: str,
) -> Dict[int, Path]:
    directory = activations_root / entity_type
    if not directory.is_dir():
        raise FileNotFoundError(f"Activation directory not found: {directory}")
    pattern = activation_pattern(entity_type)
    files: Dict[int, Path] = {}
    for path in directory.iterdir():
        match = pattern.fullmatch(path.name)
        if match:
            files[int(match.group(1))] = path
    if not files:
        raise FileNotFoundError(f"No matching layer files found in: {directory}")
    return files


def load_activation_tensor(path: Path, expected_rows: int) -> np.ndarray:
    try:
        saved = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        saved = torch.load(path, map_location="cpu")

    if isinstance(saved, torch.Tensor):
        tensor = saved
    elif isinstance(saved, dict) and isinstance(saved.get("activations"), torch.Tensor):
        tensor = saved["activations"]
    else:
        raise TypeError(f"Unsupported activation object: {path}")

    if tensor.ndim != 2:
        raise ValueError(f"Expected a 2-D tensor in {path}, got {tuple(tensor.shape)}")
    if tensor.shape[0] != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} rows in {path}, got {tensor.shape[0]}. "
            "Check the CSV order and template count."
        )
    array = tensor.detach().cpu().float().numpy()
    if not np.isfinite(array).all():
        raise ValueError(f"Non-finite activation value found in: {path}")
    return array


def repeated_metadata(
    dataframe: pd.DataFrame,
    templates_per_molecule: int,
    group_by: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    labels = np.repeat(
        (dataframe["functional_group_position"].to_numpy(dtype=int) == 2).astype(int),
        templates_per_molecule,
    )
    molecules = np.repeat(
        dataframe["iupac_name"].to_numpy(dtype=str), templates_per_molecule
    )
    pair_ids = np.repeat(
        dataframe["pair_id"].to_numpy(dtype=str), templates_per_molecule
    )
    groups = pair_ids if group_by == "pair" else molecules
    return labels, groups, molecules, pair_ids


def make_splitter(n_splits: int):
    if StratifiedGroupKFold is not None:
        return StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=RANDOM_STATE,
        )
    print(
        "WARNING: StratifiedGroupKFold is unavailable; using GroupKFold.",
        file=sys.stderr,
    )
    return GroupKFold(n_splits=n_splits)


def safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    return float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) == 2 else np.nan


def fit_grouped_probe(
    activations: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    molecules: np.ndarray,
    pair_ids: np.ndarray,
    n_splits: int,
    max_pca_components: int,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    splitter = make_splitter(n_splits)
    oof_probability = np.full(len(labels), np.nan, dtype=float)
    oof_fold = np.full(len(labels), -1, dtype=int)
    template_fold_scores: List[float] = []
    molecule_fold_scores: List[float] = []

    for fold, (train_index, test_index) in enumerate(
        splitter.split(activations, labels, groups=groups)
    ):
        scaler = StandardScaler()
        x_train = scaler.fit_transform(activations[train_index])
        x_test = scaler.transform(activations[test_index])

        components = min(
            max_pca_components,
            x_train.shape[0] - 1,
            x_train.shape[1],
        )
        if components < 1:
            raise ValueError("At least one PCA component is required")
        pca = PCA(n_components=components, random_state=RANDOM_STATE)
        x_train = pca.fit_transform(x_train)
        x_test = pca.transform(x_test)

        classifier = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=5000,
            random_state=RANDOM_STATE,
            solver="liblinear",
        )
        classifier.fit(x_train, labels[train_index])
        probability = classifier.predict_proba(x_test)[:, 1]
        prediction = (probability >= 0.5).astype(int)
        oof_probability[test_index] = probability
        oof_fold[test_index] = fold
        template_fold_scores.append(
            balanced_accuracy_score(labels[test_index], prediction)
        )

        fold_frame = pd.DataFrame(
            {
                "molecule": molecules[test_index],
                "true_label": labels[test_index],
                "probability_position_2": probability,
            }
        )
        fold_molecules = fold_frame.groupby("molecule", as_index=False).agg(
            true_label=("true_label", "first"),
            probability_position_2=("probability_position_2", "mean"),
        )
        molecule_fold_scores.append(
            balanced_accuracy_score(
                fold_molecules["true_label"],
                (fold_molecules["probability_position_2"] >= 0.5).astype(int),
            )
        )

    if np.isnan(oof_probability).any() or (oof_fold < 0).any():
        raise RuntimeError("Cross-validation did not predict every activation row")

    template_prediction = (oof_probability >= 0.5).astype(int)
    template_metrics = {
        "template_accuracy": float(accuracy_score(labels, template_prediction)),
        "template_balanced_accuracy": float(
            balanced_accuracy_score(labels, template_prediction)
        ),
        "template_roc_auc": safe_auc(labels, oof_probability),
        "template_fold_balanced_accuracy_mean": float(np.mean(template_fold_scores)),
        "template_fold_balanced_accuracy_sd": float(
            np.std(template_fold_scores, ddof=1)
        ),
    }

    prediction_frame = pd.DataFrame(
        {
            "molecule": molecules,
            "pair_id": pair_ids,
            "true_label": labels,
            "probability_position_2": oof_probability,
            "fold": oof_fold,
        }
    )
    molecule_predictions = prediction_frame.groupby("molecule", as_index=False).agg(
        pair_id=("pair_id", "first"),
        true_label=("true_label", "first"),
        probability_position_2=("probability_position_2", "mean"),
        fold=("fold", "first"),
    )
    molecule_predictions["true_position"] = molecule_predictions["true_label"] + 1
    molecule_predictions["predicted_position"] = np.where(
        molecule_predictions["probability_position_2"] >= 0.5, 2, 1
    )

    molecule_true = molecule_predictions["true_label"].to_numpy(dtype=int)
    molecule_probability = molecule_predictions["probability_position_2"].to_numpy(
        dtype=float
    )
    molecule_predicted = (molecule_probability >= 0.5).astype(int)
    metrics = {
        **template_metrics,
        "molecule_accuracy": float(accuracy_score(molecule_true, molecule_predicted)),
        "molecule_balanced_accuracy": float(
            balanced_accuracy_score(molecule_true, molecule_predicted)
        ),
        "molecule_roc_auc": safe_auc(molecule_true, molecule_probability),
        "molecule_fold_balanced_accuracy_mean": float(np.mean(molecule_fold_scores)),
        "molecule_fold_balanced_accuracy_sd": float(
            np.std(molecule_fold_scores, ddof=1)
        ),
    }
    return metrics, molecule_predictions


def shared_layers(
    condition_files: Dict[str, Dict[int, Path]], requested: Optional[Iterable[int]]
) -> List[int]:
    sets = [set(files) for files in condition_files.values()]
    available = set.intersection(*sets)
    if not available:
        raise FileNotFoundError("The conditions have no shared activation layers")
    if requested is None or len(list(requested)) == 0:
        return sorted(available)
    selected = list(dict.fromkeys(requested))
    missing = [layer for layer in selected if layer not in available]
    if missing:
        raise FileNotFoundError(
            "Requested layer(s) missing from at least one condition: "
            + ", ".join(map(str, missing))
        )
    return selected


def save_accuracy_plot(results: pd.DataFrame, output_path: Path) -> None:
    fig, axis = plt.subplots(figsize=(9, 5.5))
    for condition in CONDITIONS:
        subset = results.loc[results["condition"].eq(condition)].sort_values("layer")
        axis.plot(
            subset["layer"],
            subset["molecule_balanced_accuracy"],
            marker="o",
            markersize=3.5,
            linewidth=1.6,
            label=condition,
        )
    axis.axhline(BASELINE, color="gray", linestyle="--", linewidth=1.4, label="50% baseline")
    axis.set_xlabel("Layer index")
    axis.set_ylabel("Molecule-level balanced accuracy")
    axis.set_ylim(0, 1.02)
    axis.set_title("Functional-group position probe across prompt conditions")
    axis.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    axis.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if args.templates_per_molecule < 1:
        raise ValueError("templates-per-molecule must be at least 1")
    if args.folds < 2:
        raise ValueError("folds must be at least 2")
    if args.pca_components < 1:
        raise ValueError("pca-components must be at least 1")

    csv_path = Path(args.csv).expanduser().resolve()
    activations_root = Path(args.activations_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    dataframe = load_dataset(csv_path)
    expected_rows = len(dataframe) * args.templates_per_molecule
    labels, groups, molecules, pair_ids = repeated_metadata(
        dataframe, args.templates_per_molecule, args.group_by
    )

    condition_files = {
        condition: discover_condition_files(activations_root, entity_type)
        for condition, entity_type in CONDITIONS.items()
    }
    layers = shared_layers(condition_files, args.layers)

    print(f"Dataset: {csv_path}")
    print(f"Molecules: {len(dataframe)} (12 at each position)")
    print(f"Templates per molecule: {args.templates_per_molecule}")
    print(f"Expected rows per activation tensor: {expected_rows}")
    print(f"Cross-validation grouping: {args.group_by}")
    print(f"Shared layers: {layers}")

    # Load one layer per condition during validation to catch row-count problems.
    for condition, files in condition_files.items():
        first_layer = layers[0]
        sample = load_activation_tensor(files[first_layer], expected_rows)
        print(f"Validated {condition}: layer {first_layer}, shape {sample.shape}")
    if args.validate_only:
        print("Validation finished; probes were not fitted.")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    result_rows: List[Dict[str, object]] = []
    prediction_frames: List[pd.DataFrame] = []

    for condition, files in condition_files.items():
        print(f"\nCondition: {condition}")
        for layer in layers:
            activations = load_activation_tensor(files[layer], expected_rows)
            metrics, molecule_predictions = fit_grouped_probe(
                activations=activations,
                labels=labels,
                groups=groups,
                molecules=molecules,
                pair_ids=pair_ids,
                n_splits=args.folds,
                max_pca_components=args.pca_components,
            )
            result_rows.append(
                {
                    "condition": condition,
                    "entity_type": CONDITIONS[condition],
                    "layer": layer,
                    "group_by": args.group_by,
                    "n_molecules": len(dataframe),
                    **metrics,
                }
            )
            molecule_predictions.insert(0, "layer", layer)
            molecule_predictions.insert(0, "condition", condition)
            prediction_frames.append(molecule_predictions)
            print(
                f"  layer {layer:>2}: molecule balanced accuracy "
                f"{metrics['molecule_balanced_accuracy']:.3f}"
            )

    results = pd.DataFrame(result_rows).sort_values(["condition", "layer"])
    predictions = pd.concat(prediction_frames, ignore_index=True)
    best_layers = (
        results.sort_values(
            ["condition", "molecule_balanced_accuracy", "layer"],
            ascending=[True, False, True],
        )
        .groupby("condition", as_index=False)
        .first()
    )

    results_path = output_dir / "position_probe_layer_results.csv"
    best_path = output_dir / "position_probe_best_layers.csv"
    predictions_path = output_dir / "position_probe_molecule_predictions.csv"
    plot_path = output_dir / "position_probe_accuracy_by_layer.png"
    results.to_csv(results_path, index=False)
    best_layers.to_csv(best_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    save_accuracy_plot(results, plot_path)

    print("\nBest layer per condition:")
    print(
        best_layers[
            ["condition", "layer", "molecule_balanced_accuracy", "molecule_roc_auc"]
        ].to_string(index=False)
    )
    print(f"Saved: {results_path}")
    print(f"Saved: {best_path}")
    print(f"Saved: {predictions_path}")
    print(f"Saved: {plot_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

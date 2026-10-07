#!/usr/bin/env python3
"""Plot existing functional-group-prompt activations by group and position.

Each model layer is reduced once with PCA followed by t-SNE.  The resulting
coordinates are then displayed in two panels: the first is colored by
functional-group identity, and the second by functional-group position.  This
keeps the geometry identical between panels, so only the labels/colors change.

This script does not run Llama and does not require new position prompts.  It
uses the already extracted ``functional_group`` activations.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score


ENTITY_TYPE = "functional_group"
PROMPT_NAME = "10_templates"
ID_COLUMN = "iupac_name"
FUNCTIONAL_GROUP_COLUMN = "functional_group"
POSITION_COLUMN = "functional_group_position"
DEFAULT_LAYERS = (0, 16, 31)
POSITION_TEST_GROUPS = {"alcohol", "amine", "thiol"}
POINT_SIZE = 20
RANDOM_STATE = 42


def default_project_root() -> Path:
    script_dir = Path(__file__).resolve().parent
    return script_dir.parent if script_dir.name == "TSNE_code" else script_dir


def parse_args() -> argparse.Namespace:
    project_root = default_project_root()
    parser = argparse.ArgumentParser(
        description=(
            "Run t-SNE on existing functional-group-prompt activations and "
            "color identical coordinates by functional group and position."
        )
    )
    parser.add_argument(
        "--csv",
        default=str(project_root / "functional_group_dataset_with_position.csv"),
        help="Dataset containing functional_group_position",
    )
    parser.add_argument(
        "--activations-dir",
        default=str(
            project_root
            / "activation_datasets_functional_groups_2"
            / "meta-llama-Meta-Llama-3.1-8B"
            / ENTITY_TYPE
        ),
        help="Directory containing functional_group layer .pt files",
    )
    parser.add_argument(
        "--output-dir",
        default=str(project_root / "Results" / "results_tsne_fg_position_8B"),
        help="Directory for output PNG files",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=list(DEFAULT_LAYERS),
        help="Zero-based model layers to plot (default: 0 16 31)",
    )
    parser.add_argument(
        "--templates-per-molecule",
        type=int,
        default=10,
        help="Number of functional-group prompt templates per molecule",
    )
    parser.add_argument(
        "--model-label",
        default="Llama-3.1-8B",
        help="Model name displayed in plot titles",
    )
    parser.add_argument(
        "--perplexity",
        type=float,
        default=30.0,
        help="Requested t-SNE perplexity; reduced automatically if necessary",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate paths, labels, and expected row counts without running t-SNE",
    )
    return parser.parse_args()


def load_dataset(
    csv_path: Path,
    templates_per_molecule: int,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    if templates_per_molecule < 1:
        raise ValueError("templates_per_molecule must be at least 1")

    dataframe = pd.read_csv(csv_path)
    required = {ID_COLUMN, FUNCTIONAL_GROUP_COLUMN, POSITION_COLUMN}
    missing = sorted(required - set(dataframe.columns))
    if missing:
        raise ValueError(
            f"Dataset is missing required column(s): {', '.join(missing)}"
        )
    if dataframe.empty:
        raise ValueError("Dataset contains no molecules")
    if dataframe[ID_COLUMN].isna().any() or dataframe[ID_COLUMN].duplicated().any():
        raise ValueError(f"'{ID_COLUMN}' must contain unique, nonmissing names")

    group_values = dataframe[FUNCTIONAL_GROUP_COLUMN].astype("string").str.strip()
    if group_values.isna().any() or group_values.eq("").any():
        raise ValueError(f"'{FUNCTIONAL_GROUP_COLUMN}' contains a missing label")

    raw_position = dataframe[POSITION_COLUMN]
    position_values = pd.to_numeric(raw_position, errors="coerce")
    invalid_nonblank = (
        raw_position.notna()
        & raw_position.astype("string").str.strip().ne("")
        & position_values.isna()
    )
    if invalid_nonblank.any():
        bad_names = dataframe.loc[invalid_nonblank, ID_COLUMN].tolist()
        raise ValueError(
            f"Non-numeric position label(s) found for: {', '.join(bad_names)}"
        )
    finite_positions = position_values.dropna().to_numpy(dtype=float)
    if finite_positions.size and (
        (finite_positions < 1).any()
        or not np.allclose(finite_positions, np.round(finite_positions))
    ):
        raise ValueError("Functional-group positions must be positive integers")

    repeated_features = {
        FUNCTIONAL_GROUP_COLUMN: np.repeat(
            group_values.to_numpy(dtype=str), templates_per_molecule
        ),
        POSITION_COLUMN: np.repeat(
            position_values.to_numpy(dtype=float), templates_per_molecule
        ),
    }
    return dataframe, repeated_features


def build_layer_pattern() -> re.Pattern[str]:
    return re.compile(
        rf"{re.escape(ENTITY_TYPE)}\.last\."
        rf"{re.escape(PROMPT_NAME)}\.layer_(\d+)\.pt"
    )


def get_layer_files(directory: Path) -> List[Tuple[int, Path]]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Activation directory not found: {directory}")
    pattern = build_layer_pattern()
    files: List[Tuple[int, Path]] = []
    for path in directory.iterdir():
        match = pattern.fullmatch(path.name)
        if match:
            files.append((int(match.group(1)), path))
    return sorted(files, key=lambda item: item[0])


def load_activations(path: Path, expected_rows: int) -> np.ndarray:
    try:
        saved = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        saved = torch.load(path, map_location="cpu")

    if isinstance(saved, torch.Tensor):
        tensor = saved
    elif isinstance(saved, dict) and isinstance(saved.get("activations"), torch.Tensor):
        tensor = saved["activations"]
    else:
        raise TypeError(f"Unsupported activation object in {path}")

    if tensor.ndim != 2:
        raise ValueError(
            f"Expected a 2-D activation tensor in {path}, got {tuple(tensor.shape)}"
        )
    if tensor.shape[0] != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} activation rows in {path}, got "
            f"{tensor.shape[0]}. The CSV row order/count and the template count "
            "must match the extraction run."
        )
    activations = tensor.detach().cpu().float().numpy()
    if not np.isfinite(activations).all():
        raise ValueError(f"Activation tensor contains NaN or infinity: {path}")
    return activations


def perform_pca(activations: np.ndarray) -> Tuple[np.ndarray, PCA]:
    components = min(50, activations.shape[0] - 1, activations.shape[1])
    if components < 2:
        raise ValueError("At least two PCA components are required")
    pca = PCA(n_components=components, random_state=RANDOM_STATE)
    return pca.fit_transform(activations), pca


def perform_tsne(pca_data: np.ndarray, requested_perplexity: float) -> np.ndarray:
    if requested_perplexity <= 0:
        raise ValueError("perplexity must be greater than zero")
    perplexity = min(float(requested_perplexity), float(pca_data.shape[0] - 1))
    common = {
        "n_components": 2,
        "random_state": RANDOM_STATE,
        "perplexity": perplexity,
        "init": "pca",
        "learning_rate": "auto",
    }
    try:
        model = TSNE(max_iter=1000, **common)
    except TypeError:
        # Compatibility with older scikit-learn versions on some HCC nodes.
        model = TSNE(n_iter=1000, **common)
    return model.fit_transform(pca_data)


def category_sort_key(label: str) -> Tuple[int, object]:
    match = re.fullmatch(r"Position (\d+)", label)
    return (0, int(match.group(1))) if match else (1, label.casefold())


def prepared_labels(feature: str, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if feature == POSITION_COLUMN:
        numeric = np.asarray(values, dtype=float)
        valid = np.isfinite(numeric)
        labels = np.full(len(numeric), "", dtype=object)
        labels[valid] = [f"Position {int(value)}" for value in numeric[valid]]
        return labels, valid

    series = pd.Series(values, dtype="string").str.strip()
    valid = (series.notna() & series.ne("")).to_numpy(dtype=bool)
    labels = series.fillna("").to_numpy(dtype=str)
    return labels, valid


def colors_for_categories(categories: Sequence[str], feature: str) -> Dict[str, object]:
    ordered = sorted(set(categories), key=category_sort_key)
    if feature == POSITION_COLUMN:
        palette = ["#1F77B4", "#D62728", "#2CA02C", "#9467BD"]
        return {category: palette[index % len(palette)] for index, category in enumerate(ordered)}
    colors = plt.cm.tab20(np.linspace(0, 1, max(len(ordered), 1)))
    return {category: colors[index] for index, category in enumerate(ordered)}


def feature_title(feature: str) -> str:
    return {
        FUNCTIONAL_GROUP_COLUMN: "Colored by functional-group identity",
        POSITION_COLUMN: "Colored by functional-group position",
    }.get(feature, feature)


def plot_feature_panel(
    axis: plt.Axes,
    tsne_data: np.ndarray,
    pca_data: np.ndarray,
    feature: str,
    values: np.ndarray,
) -> None:
    labels, valid = prepared_labels(feature, values)
    invalid = ~valid
    if invalid.any():
        axis.scatter(
            tsne_data[invalid, 0],
            tsne_data[invalid, 1],
            s=POINT_SIZE,
            c="#B8B8B8",
            marker="x",
            alpha=0.55,
            label=f"N/A ({int(invalid.sum())} points)",
        )

    valid_labels = labels[valid]
    color_map = colors_for_categories(valid_labels, feature)
    for category in sorted(color_map, key=category_sort_key):
        category_mask = valid & (labels == category)
        axis.scatter(
            tsne_data[category_mask, 0],
            tsne_data[category_mask, 1],
            s=POINT_SIZE,
            c=[color_map[category]],
            marker="o",
            alpha=0.72,
            label=f"{category} ({int(category_mask.sum())} points)",
        )

    unique_labels = np.unique(valid_labels)
    if len(unique_labels) > 1 and valid.sum() > len(unique_labels):
        encoded = pd.factorize(valid_labels, sort=True)[0]
        try:
            silhouette = silhouette_score(pca_data[valid], encoded)
            axis.text(
                0.98,
                0.02,
                f"PCA-space silhouette: {silhouette:.2f}",
                transform=axis.transAxes,
                ha="right",
                va="bottom",
                fontsize=9,
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.75},
            )
        except ValueError:
            pass

    axis.set_title(feature_title(feature), fontsize=12)
    axis.set_aspect("equal", adjustable="datalim")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=7 if feature == FUNCTIONAL_GROUP_COLUMN else 9,
        frameon=True,
    )


def plot_layer(
    tsne_data: np.ndarray,
    pca_data: np.ndarray,
    features: Dict[str, np.ndarray],
    layer: int,
    model_label: str,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(17, 7))
    for axis, feature in zip(
        axes, (FUNCTIONAL_GROUP_COLUMN, POSITION_COLUMN)
    ):
        plot_feature_panel(axis, tsne_data, pca_data, feature, features[feature])

    fig.suptitle(
        f"Functional-group-prompt activations — layer {layer} ({model_label})\n"
        "Identical t-SNE coordinates in both panels",
        fontsize=15,
    )
    fig.subplots_adjust(left=0.04, right=0.88, bottom=0.06, top=0.86, wspace=0.42)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    csv_path = Path(args.csv).expanduser().resolve()
    activations_dir = Path(args.activations_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    # Load labels for the complete activation dataset.
    dataframe, all_features = load_dataset(
        csv_path,
        args.templates_per_molecule,
    )
    
    # The activation files still contain all 92 molecules.
    expected_rows = len(dataframe) * args.templates_per_molecule
    
    position_values = pd.to_numeric(
        dataframe[POSITION_COLUMN],
        errors="coerce",
    )
    
    # Keep only alcohol, amine, and thiol with positions 1 or 2.
    molecule_mask = (
        dataframe[FUNCTIONAL_GROUP_COLUMN].isin(POSITION_TEST_GROUPS)
        & position_values.isin([1, 2])
    )
    
    molecule_indices = np.flatnonzero(molecule_mask.to_numpy())
    
    if molecule_indices.size == 0:
        raise ValueError("No molecules matched the position-test subset")
    
    # Convert molecule indices into matching activation-row indices.
    # Every molecule has 10 consecutive activation rows.
    activation_indices = np.concatenate([
        np.arange(
            molecule_index * args.templates_per_molecule,
            (molecule_index + 1) * args.templates_per_molecule,
        )
        for molecule_index in molecule_indices
    ])
    
    selected_dataframe = dataframe.iloc[molecule_indices].copy()
    
    # Select the same rows from the repeated plot labels.
    features = {
        feature_name: feature_values[activation_indices]
        for feature_name, feature_values in all_features.items()
    }
    
    position_counts = (
        pd.to_numeric(
            selected_dataframe[POSITION_COLUMN],
            errors="coerce",
        )
        .value_counts(dropna=False)
        .sort_index()
    )

    layer_files = get_layer_files(activations_dir)
    requested_layers = list(dict.fromkeys(args.layers))
    available = {layer: path for layer, path in layer_files}
    missing_layers = [layer for layer in requested_layers if layer not in available]

    print(f"Dataset: {csv_path}")
    print(f"Total molecules in activation dataset: {len(dataframe)}")
    print(f"Selected molecules: {len(selected_dataframe)}")
    print(f"Templates per molecule: {args.templates_per_molecule}")
    print(f"Full activation rows per layer: {expected_rows}")
    print(f"Selected activation rows: {len(activation_indices)}")
    print(f"Activation directory: {activations_dir}")
    print("Functional-group position counts (molecules):")
    for value, count in position_counts.items():
        label = "N/A" if pd.isna(value) else str(int(value))
        print(f"  {label}: {int(count)}")
    print(f"Requested layers: {requested_layers}")

    if missing_layers:
        raise FileNotFoundError(
            "Missing requested activation layer(s): "
            + ", ".join(map(str, missing_layers))
        )
    if args.validate_only:
        print("Validation finished; t-SNE was not run.")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    for layer in requested_layers:
        activation_path = available[layer]
        print(f"\nProcessing layer {layer}: {activation_path.name}")
        # Validate and load the complete 920-row activation tensor first.
        activations = load_activations(activation_path, expected_rows)
        
        # Then select the 240 rows belonging to the balanced subset.
        activations = activations[activation_indices]
        
        pca_data, pca = perform_pca(activations)
        print(
            "PCA retained variance: "
            f"{pca.explained_variance_ratio_.sum():.3f} "
            f"({pca.n_components_} components)"
        )
        tsne_data = perform_tsne(pca_data, args.perplexity)
        output_path = output_dir / (
            f"layer_{layer}_tsne_functional_group_vs_position.png"
        )
        plot_layer(
            tsne_data,
            pca_data,
            features,
            layer,
            args.model_label,
            output_path,
        )
        print(f"Saved: {output_path}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

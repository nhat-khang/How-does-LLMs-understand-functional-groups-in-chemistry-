"""Indirect-recall probes for the functional-group project.

An experiment uses activations produced by prompts about one property
(the source) to predict a different CSV property (the target).  Examples:

    molecular-weight prompt activations -> functional-group label
    functional-group prompt activations -> molecular weight

This version uses one-molecule functional-group support. For each functional
group in turn, one molecule (and all of its prompt templates) is included in
training, while the other molecules from that group are tested. All molecules
from other functional groups remain in training. The support molecule is
rotated to avoid basing the conclusion on one arbitrary choice.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR


CLASSIFICATION = "classification"
REGRESSION = "regression"
VALID_TARGET_TYPES = {CLASSIFICATION, REGRESSION}


@dataclass(frozen=True)
class SourceConfig:
    entity_type: str
    prompt_name: str
    templates_per_molecule: int
    extraction_data_file: Optional[str]


@dataclass(frozen=True)
class Experiment:
    name: str
    source_entity: str
    target_column: str
    target_type: str
    prompt_name: str
    templates_per_molecule: int
    activation_path_template: str
    num_layers: int
    model_name: str


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return data


def resolve_path(value: str, config_dir: Path) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(value)))
    return path if path.is_absolute() else (config_dir / path).resolve()


def load_source_configs(extraction_config_path: Path) -> Dict[str, SourceConfig]:
    config = load_yaml(extraction_config_path)
    try:
        entries = config["extraction"]["entities"]
    except KeyError as exc:
        raise ValueError(
            f"{extraction_config_path} must contain extraction.entities"
        ) from exc

    sources: Dict[str, SourceConfig] = {}
    for entry in entries:
        entity_type = str(entry["entity_type"])
        templates = entry.get("templates", [])
        if not templates:
            raise ValueError(
                f"Source entity '{entity_type}' has no prompt templates in "
                f"{extraction_config_path}"
            )
        if entity_type in sources:
            raise ValueError(f"Duplicate source entity '{entity_type}'")
        sources[entity_type] = SourceConfig(
            entity_type=entity_type,
            prompt_name=str(entry["prompt_name"]),
            templates_per_molecule=len(templates),
            extraction_data_file=entry.get("data_file"),
        )
    return sources


def sanitize_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def build_experiments(
    config: Dict[str, Any],
    source_configs: Dict[str, SourceConfig],
    activation_base_dir: Path,
    selected_names: Optional[Sequence[str]] = None,
) -> List[Experiment]:
    model_config = config.get("model", {})
    default_model_name = str(model_config.get("name", "model"))
    default_num_layers = int(model_config.get("num_layers", 0))
    if default_num_layers < 1:
        raise ValueError("model.num_layers must be a positive integer")

    allow_direct = bool(config.get("analysis", {}).get("allow_direct_pairs", False))
    selected = set(selected_names or [])
    experiments: List[Experiment] = []

    for raw in config.get("experiments", []):
        if not raw.get("enabled", True):
            continue

        source_entity = str(raw["source_entity"])
        target_column = str(raw["target_column"])
        name = str(raw.get("name", f"{source_entity}_to_{target_column}"))
        if selected and name not in selected:
            continue
        if source_entity not in source_configs:
            available = ", ".join(sorted(source_configs))
            raise ValueError(
                f"Experiment '{name}' uses unknown source_entity "
                f"'{source_entity}'. Available sources: {available}"
            )
        if source_entity == target_column and not allow_direct:
            raise ValueError(
                f"Experiment '{name}' maps '{source_entity}' to itself. That is "
                "direct recall. Use two different properties, or set "
                "analysis.allow_direct_pairs: true intentionally."
            )

        target_type = str(raw.get("target_type", "")).lower()
        if target_type not in VALID_TARGET_TYPES:
            raise ValueError(
                f"Experiment '{name}' target_type must be one of "
                f"{sorted(VALID_TARGET_TYPES)}, not '{target_type}'"
            )

        source = source_configs[source_entity]
        prompt_name = str(raw.get("prompt_name", source.prompt_name))
        templates_per_molecule = int(
            raw.get("templates_per_molecule", source.templates_per_molecule)
        )
        num_layers = int(raw.get("num_layers", default_num_layers))
        model_name = str(raw.get("model_name", default_model_name))

        custom_template = raw.get("activation_path_template")
        if custom_template:
            activation_template = str(
                resolve_path(str(custom_template), activation_base_dir)
                if not Path(os.path.expanduser(str(custom_template))).is_absolute()
                else Path(os.path.expandvars(os.path.expanduser(str(custom_template))))
            )
        else:
            activation_template = str(
                activation_base_dir
                / source_entity
                / (
                    f"{source_entity}.last.{prompt_name}."
                    "layer_{layer}.pt"
                )
            )

        if "{layer}" not in activation_template:
            raise ValueError(
                f"Experiment '{name}' activation path must contain '{{layer}}'"
            )
        if templates_per_molecule < 1 or num_layers < 1:
            raise ValueError(
                f"Experiment '{name}' has an invalid template or layer count"
            )

        experiments.append(
            Experiment(
                name=name,
                source_entity=source_entity,
                target_column=target_column,
                target_type=target_type,
                prompt_name=prompt_name,
                templates_per_molecule=templates_per_molecule,
                activation_path_template=activation_template,
                num_layers=num_layers,
                model_name=model_name,
            )
        )

    if selected:
        found = {experiment.name for experiment in experiments}
        missing = selected - found
        if missing:
            raise ValueError(
                "Requested experiment(s) were not found or enabled: "
                + ", ".join(sorted(missing))
            )
    if not experiments:
        raise ValueError("No experiments are enabled in the configuration")
    return experiments


def load_target_by_molecule(
    dataframe: pd.DataFrame,
    target_column: str,
    target_type: str,
) -> Tuple[np.ndarray, np.ndarray]:
    if target_column not in dataframe.columns:
        raise ValueError(
            f"Target column '{target_column}' is absent from the dataset. "
            f"Available columns: {', '.join(dataframe.columns)}"
        )

    raw = dataframe[target_column]
    if target_type == REGRESSION:
        values = pd.to_numeric(raw, errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(values)
    else:
        stripped = raw.astype("string").str.strip()
        # Pandas' nullable string dtype can produce a BooleanArray containing
        # pd.NA.  Convert it explicitly to a plain NumPy bool array before it
        # is used to index NumPy target arrays.
        valid = (
            stripped.notna() & stripped.ne("")
        ).fillna(False).to_numpy(dtype=bool)
        values = stripped.fillna("").to_numpy(dtype=str)

    if not valid.any():
        raise ValueError(
            f"Target column '{target_column}' has no valid {target_type} values"
        )
    return values, valid


def load_split_groups_by_molecule(
    dataframe: pd.DataFrame,
    split_by: str,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Return one cross-validation group label per molecule.

    ``split_by: molecule`` gives each molecule its own group, preserving the
    original behavior.  Any other value is interpreted as a dataset column,
    such as ``functional_group``.  In that mode, all molecules sharing the
    same column value are held out together.
    """
    split_by = str(split_by).strip()
    if not split_by:
        raise ValueError("analysis.split_by must not be empty")

    if split_by.lower() == "molecule":
        values = np.arange(len(dataframe), dtype=int)
        valid = np.ones(len(dataframe), dtype=bool)
        return values, valid, "molecule"

    if split_by not in dataframe.columns:
        raise ValueError(
            f"analysis.split_by='{split_by}' is neither 'molecule' nor a "
            f"dataset column. Available columns: {', '.join(dataframe.columns)}"
        )

    raw = dataframe[split_by].astype("string").str.strip()
    valid = (raw.notna() & raw.ne("")).fillna(False).to_numpy(dtype=bool)
    values = raw.fillna("").to_numpy(dtype=str)
    if not valid.any():
        raise ValueError(f"Split column '{split_by}' has no valid values")
    return values, valid, split_by


def load_activation(path: Path, expected_rows: int) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")

    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Expected a tensor in {path}, got {type(value).__name__}")
    if value.ndim != 2:
        raise ValueError(
            f"Expected a 2-D [prompt, hidden_dimension] tensor in {path}; "
            f"got shape {tuple(value.shape)}"
        )
    if value.shape[0] != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} activation rows in {path}, "
            f"got {value.shape[0]}. Check the dataset order/count and the "
            "templates_per_molecule value."
        )
    array = value.detach().cpu().float().numpy()
    if not np.isfinite(array).all():
        raise ValueError(f"Activation file contains NaN or infinity: {path}")
    return array


def build_one_support_splits(
    groups: np.ndarray,
    molecule_ids: np.ndarray,
    n_splits: int = 5,
    n_support_repeats: int = 20,
    base_seed: int = 42,
):
    """
    Five-fold functional-group-held-out evaluation with one support
    molecule from EACH eligible held-out functional group.

    For each fold:
        base training =
            all molecules outside the held-out functional groups

        one-shot training =
            base training
            + one molecule from each eligible held-out functional group

        testing =
            remaining molecules from those supported held-out groups
    """

    splits = []

    effective_n_splits = min(
        n_splits,
        len(np.unique(groups)),
    )

    gkf = GroupKFold(
        n_splits=effective_n_splits
    )

    dummy_X = np.zeros(len(groups))

    for fold_ix, (base_train_idx, heldout_idx) in enumerate(
        gkf.split(
            dummy_X,
            groups=groups,
        )
    ):

        heldout_groups = np.unique(
            groups[heldout_idx]
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
                    groups[heldout_idx] == heldout_group
                ]

                group_molecules = np.unique(
                    molecule_ids[group_rows]
                )

                # Need at least:
                # 1 support molecule + 1 test molecule
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

            support_mask = np.isin(
                molecule_ids,
                support_molecules,
            )

            support_idx = np.flatnonzero(
                support_mask
            )

            train_idx = np.unique(
                np.concatenate([
                    base_train_idx,
                    support_idx,
                ])
            )

            heldout_mask = np.zeros(
                len(groups),
                dtype=bool,
            )

            heldout_mask[heldout_idx] = True

            supported_group_mask = np.isin(
                groups,
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

def one_support_regression_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    molecule_ids: np.ndarray,
    kernel: str,
    c_value: float,
    n_splits: int = 5,
    n_support_repeats: int = 20,
    base_seed: int = 42,
) -> Dict[str, float]:

    fold_r2: List[float] = []
    fold_mae: List[float] = []

    n_r2_skipped = 0

    for train_idx, test_idx, _, _ in build_one_support_splits(
        groups,
        molecule_ids,
        n_splits=n_splits,
        n_support_repeats=n_support_repeats,
        base_seed=base_seed,
    ):

        scaler = StandardScaler()

        X_train = scaler.fit_transform(
            X[train_idx]
        )

        X_test = scaler.transform(
            X[test_idx]
        )

        probe = SVR(
            kernel=kernel,
            C=c_value,
        )

        probe.fit(
            X_train,
            y[train_idx],
        )

        predictions = probe.predict(
            X_test
        )

        # MAE is valid for every test split.
        fold_mae.append(
            float(
                mean_absolute_error(
                    y[test_idx],
                    predictions,
                )
            )
        )

        # R2 requires variation in the test targets.
        if len(np.unique(y[test_idx])) >= 2:

            score = float(
                r2_score(
                    y[test_idx],
                    predictions,
                )
            )

            if np.isfinite(score):
                fold_r2.append(score)

            else:
                n_r2_skipped += 1

        else:
            n_r2_skipped += 1

    if not fold_mae:
        raise ValueError(
            "No valid one-support regression splits remained."
        )

    mean_r2 = (
        float(np.mean(fold_r2))
        if fold_r2
        else np.nan
    )

    std_r2 = (
        float(np.std(fold_r2))
        if fold_r2
        else np.nan
    )

    return {
        "score": mean_r2,
        "score_std": std_r2,

        "r2": mean_r2,

        "mae": float(np.mean(fold_mae)),
        "mae_std": float(np.std(fold_mae)),

        "accuracy": np.nan,
        "balanced_accuracy": np.nan,
        "macro_f1": np.nan,

        "n_mae_splits": len(fold_mae),
        "n_r2_splits": len(fold_r2),
        "n_r2_skipped": n_r2_skipped,
    }

def one_support_classification_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    molecule_ids: np.ndarray,
    kernel: str,
    c_value: float,
    class_weight: Optional[str],
) -> Dict[str, float]:
    fold_accuracy: List[float] = []
    fold_balanced_accuracy: List[float] = []
    fold_macro_f1: List[float] = []
    skipped = 0

    for train_idx, test_idx, _, _ in build_one_support_splits(
        groups, molecule_ids
    ):
        if len(np.unique(y[train_idx])) < 2:
            skipped += 1
            continue
        unseen_classes = sorted(set(y[test_idx]) - set(y[train_idx]))
        if unseen_classes:
            skipped += 1
            continue
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        probe = SVC(kernel=kernel, C=c_value, class_weight=class_weight)
        probe.fit(X_train, y[train_idx])
        predictions = probe.predict(X_test)
        fold_accuracy.append(float(accuracy_score(y[test_idx], predictions)))
        fold_balanced_accuracy.append(
            float(balanced_accuracy_score(y[test_idx], predictions))
        )
        fold_macro_f1.append(
            float(f1_score(y[test_idx], predictions, average="macro", zero_division=0))
        )

    if not fold_accuracy:
        raise ValueError("No valid one-support classification splits remained.")

    return {
        "score": float(np.mean(fold_accuracy)),
        "score_std": float(np.std(fold_accuracy)),
        "r2": np.nan,
        "mae": np.nan,
        "accuracy": float(np.mean(fold_accuracy)),
        "balanced_accuracy": float(np.mean(fold_balanced_accuracy)),
        "macro_f1": float(np.mean(fold_macro_f1)),
        "n_evaluation_splits": len(fold_accuracy),
        "n_skipped_splits": skipped,
    }


def plot_experiment(
    rows: List[Dict[str, Any]],
    experiment: Experiment,
    output_dir: Path,
    majority_baseline: Optional[float],
    split_by: str,
    split_strategy: str,
) -> Path:
    valid_rows = [row for row in rows if np.isfinite(row["score"])]
    layers = [row["layer"] for row in valid_rows]
    scores = [row["score"] for row in valid_rows]
    deviations = [row["score_std"] for row in valid_rows]

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(layers, scores, color="#1f77b4", marker="o", markersize=3)
    axis.fill_between(
        layers,
        np.asarray(scores) - np.asarray(deviations),
        np.asarray(scores) + np.asarray(deviations),
        color="#1f77b4",
        alpha=0.15,
        label="±1 fold SD",
    )
    if experiment.target_type == CLASSIFICATION and majority_baseline is not None:
        axis.axhline(
            majority_baseline,
            color="gray",
            linestyle="--",
            label=f"Majority baseline ({majority_baseline:.3f})",
        )
        axis.set_ylabel("one-support accuracy")
        axis.set_ylim(0.0, 1.0)
    else:
        axis.axhline(0.0, color="gray", linestyle="--", linewidth=1)
        axis.set_ylabel("one-support $R^2$")

    axis.set_xlabel("Layer index")
    axis.set_title(
        f"Indirect recall: {experiment.source_entity} prompts → "
        f"{experiment.target_column}\n{experiment.model_name}; "
        f"{split_strategy}"
    )
    axis.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
    axis.legend(loc="best", fontsize=8)
    figure.tight_layout()
    output_path = output_dir / f"{sanitize_name(experiment.name)}.png"
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return output_path


def write_csv(path: Path, rows: Iterable[Dict[str, Any]], fieldnames: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_validation(
    dataframe: pd.DataFrame,
    dataset_path: Path,
    extraction_config_path: Path,
    activation_base_dir: Path,
    experiments: Sequence[Experiment],
    split_by: str,
    split_values: np.ndarray,
    valid_split_molecules: np.ndarray,
    split_strategy: str,
) -> None:
    print("\nConfiguration validation")
    print(f"  Dataset:          {dataset_path} ({len(dataframe)} molecules)")
    print(f"  Extraction YAML:  {extraction_config_path}")
    print(f"  Activation base:  {activation_base_dir}")
    print(
        f"  CV split unit:     {split_by} "
        f"({len(np.unique(split_values[valid_split_molecules]))} groups)"
    )
    print(f"  Split strategy:    {split_strategy}")
    for experiment in experiments:
        values, valid = load_target_by_molecule(
            dataframe, experiment.target_column, experiment.target_type
        )
        first_path = Path(experiment.activation_path_template.format(layer=0))
        valid_values = values[valid]
        if experiment.target_type == CLASSIFICATION:
            detail = f"{len(np.unique(valid_values))} classes"
        else:
            detail = f"{valid.sum()} numeric molecule values"
        print(f"\n  [{experiment.name}]")
        print(
            f"    {experiment.source_entity} prompts -> "
            f"{experiment.target_column} ({experiment.target_type}; {detail})"
        )
        print(
            f"    {experiment.templates_per_molecule} templates/molecule; "
            f"expected activation rows: "
            f"{len(dataframe) * experiment.templates_per_molecule}"
        )
        print(f"    Layer-0 file: {first_path}")
        print(f"    Layer-0 exists: {'yes' if first_path.exists() else 'no'}")


def run(config_path: Path, args: argparse.Namespace) -> int:
    config_path = config_path.resolve()
    config_dir = config_path.parent
    config = load_yaml(config_path)
    paths = config.get("paths", {})

    dataset_path = resolve_path(
        args.dataset or paths.get("dataset_file", "functional_group_dataset_2.csv"),
        config_dir,
    )
    extraction_config_path = resolve_path(
        args.extraction_config
        or paths.get("extraction_config", "config_extract_activation_fg_8B.yaml"),
        config_dir,
    )
    activation_base_dir = resolve_path(
        args.activation_base_dir
        or paths.get("activation_base_dir", "activation_datasets_functional_groups_2"),
        config_dir,
    )
    output_dir = resolve_path(
        args.output_dir or paths.get("output_dir", "Results/indirect_recall_fg"),
        config_dir,
    )

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    if not extraction_config_path.exists():
        raise FileNotFoundError(
            f"Extraction configuration not found: {extraction_config_path}"
        )

    dataframe = pd.read_csv(dataset_path)
    source_configs = load_source_configs(extraction_config_path)
    experiments = build_experiments(
        config,
        source_configs,
        activation_base_dir,
        selected_names=args.experiments,
    )

    analysis = config.get("analysis", {})

    split_by = str(
        analysis.get(
            "split_by",
            "functional_group",
        )
    )

    split_strategy = str(
        analysis.get(
            "split_strategy",
            "five_fold_one_support_per_held_out_functional_group",
        )
    ).strip()

    if split_strategy != "five_fold_one_support_per_held_out_functional_group":
        raise ValueError(
            "This script requires analysis.split_strategy: "
            "five_fold_one_support_per_held_out_functional_group"
        )

    support_count = int(
        analysis.get(
            "support_molecules_per_group",
            1,
        )
    )

    if support_count != 1:
        raise ValueError(
            "This script currently requires "
            "analysis.support_molecules_per_group: 1"
        )

    n_splits = int(
        analysis.get(
            "n_splits",
            5,
        )
    )

    n_support_repeats = int(
        analysis.get(
            "n_support_repeats",
            20,
        )
    )

    support_random_seed = int(
        analysis.get(
            "support_random_seed",
            42,
        )
    )
    if split_by.lower() == "molecule":
        raise ValueError(
            "One-support evaluation requires analysis.split_by to name a "
            "dataset group column, such as functional_group."
        )
    split_values, valid_split_molecules, split_by = load_split_groups_by_molecule(
        dataframe, split_by
    )

    print_validation(
        dataframe,
        dataset_path,
        extraction_config_path,
        activation_base_dir,
        experiments,
        split_by,
        split_values,
        valid_split_molecules,
        split_strategy,
    )
    if args.validate_only:
        print("\nValidation finished; probes were not run.")
        return 0

    regression = config.get("regression_probe", {})
    classification = config.get("classification_probe", {})
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, Any]] = []
    best_rows: List[Dict[str, Any]] = []

    for experiment in experiments:
        print(
            f"\n=== {experiment.name}: {experiment.source_entity} prompts -> "
            f"{experiment.target_column} ==="
        )
        molecule_values, valid_targets = load_target_by_molecule(
            dataframe, experiment.target_column, experiment.target_type
        )
        valid_molecules = valid_targets & valid_split_molecules
        all_targets = np.repeat(
            molecule_values, experiment.templates_per_molecule
        )
        all_split_groups = np.repeat(
            split_values, experiment.templates_per_molecule
        )
        all_molecule_ids = np.repeat(
            np.arange(len(dataframe)), experiment.templates_per_molecule
        )
        valid_rows = np.repeat(valid_molecules, experiment.templates_per_molecule)
        expected_rows = len(dataframe) * experiment.templates_per_molecule

        target_dropped = int((~valid_targets).sum())
        split_dropped = int((valid_targets & ~valid_split_molecules).sum())
        if target_dropped:
            print(
                f"Dropping {target_dropped} molecule(s) without a valid "
                f"'{experiment.target_column}' target."
            )
        if split_dropped:
            print(
                f"Dropping {split_dropped} additional molecule(s) without a "
                f"valid '{split_by}' split label."
            )

        y = all_targets[valid_rows]
        groups = all_split_groups[valid_rows]
        molecule_ids = all_molecule_ids[valid_rows]
        available_splits = build_one_support_splits(
            groups,
            molecule_ids,
            n_splits=n_splits,
            n_support_repeats=n_support_repeats,
            base_seed=support_random_seed,
        )
        if not available_splits:
            print(
                "  No split group has at least two valid molecules; skipping."
            )
            continue
        print(
            f"  Five-fold one-support evaluation: "
            f"{len(available_splits)} fold/support evaluations."
        )
        if experiment.target_type == REGRESSION:
            y = y.astype(float)
            majority_baseline = None
            primary_metric = "r2"
        else:
            y = y.astype(str)
            molecule_class_values = molecule_values[valid_molecules].astype(str)
            counts = pd.Series(molecule_class_values).value_counts()
            majority_baseline = float(counts.iloc[0] / counts.sum())
            primary_metric = "accuracy"

        experiment_rows: List[Dict[str, Any]] = []
        for layer in range(experiment.num_layers):
            activation_path = Path(
                experiment.activation_path_template.format(layer=layer)
            )
            try:
                X_all = load_activation(activation_path, expected_rows)
                X = X_all[valid_rows]
                if experiment.target_type == REGRESSION:
                    metrics = one_support_regression_cv(
                        X,
                        y,
                        groups,
                        molecule_ids,
                        kernel=str(
                            regression.get(
                                "kernel",
                                "linear",
                            )
                        ),
                        c_value=float(
                            regression.get(
                                "c",
                                2.0,
                            )
                        ),
                        n_splits=n_splits,
                        n_support_repeats=n_support_repeats,
                        base_seed=support_random_seed,
                    )
                else:
                    raw_class_weight = classification.get("class_weight", "balanced")
                    class_weight = None if raw_class_weight in (None, "none") else str(raw_class_weight)
                    metrics = one_support_classification_cv(
                        X,
                        y,
                        groups,
                        molecule_ids,
                        kernel=str(classification.get("kernel", "linear")),
                        c_value=float(classification.get("c", 2.0)),
                        class_weight=class_weight,
                    )
            except FileNotFoundError:
                print(f"  Layer {layer}: missing {activation_path}")
                continue
            except ValueError as exc:
                print(f"  Layer {layer}: {exc}")
                continue

            row: Dict[str, Any] = {
                "experiment": experiment.name,
                "model": experiment.model_name,
                "source_entity": experiment.source_entity,
                "target_column": experiment.target_column,
                "target_type": experiment.target_type,
                "prompt_name": experiment.prompt_name,
                "templates_per_molecule": experiment.templates_per_molecule,
                "layer": layer,
                "split_by": split_by,
                "split_strategy": split_strategy,
                "support_molecules_per_group": support_count,
                "n_split_groups": len(np.unique(groups)),
                "n_molecules": len(np.unique(molecule_ids)),
                "n_prompt_activations": len(y),
                "primary_metric": primary_metric,
                **metrics,
            }
            experiment_rows.append(row)
            summary_rows.append(row)
            print(
                f"  Layer {layer:>2}: {primary_metric}={metrics['score']:.4f} "
                f"(fold SD={metrics['score_std']:.4f})"
            )

        if not experiment_rows:
            print(
                "  No activation files were analyzed. Check paths, prompt names, "
                "template counts, and layer filename format."
            )
            continue

        best = max(experiment_rows, key=lambda row: row["score"])
        best_rows.append(best.copy())
        plot_path = plot_experiment(
            experiment_rows,
            experiment,
            output_dir,
            majority_baseline,
            split_by,
            split_strategy,
        )
        print(
            f"Best layer: {best['layer']} ({primary_metric}={best['score']:.4f})"
        )
        print(f"Saved plot: {plot_path}")

    if not summary_rows:
        print("\nNo results were produced.", file=sys.stderr)
        return 2

    fieldnames = [
        "experiment",
        "model",
        "source_entity",
        "target_column",
        "target_type",
        "prompt_name",
        "templates_per_molecule",
        "layer",
        "split_by",
        "split_strategy",
        "support_molecules_per_group",
        "n_split_groups",
        "n_molecules",
        "n_prompt_activations",
        "primary_metric",
        "score",
        "score_std",

        "r2",

        "mae",
        "mae_std",

        "accuracy",
        "balanced_accuracy",
        "macro_f1",

        "n_mae_splits",
        "n_r2_splits",
        "n_r2_skipped",

        "n_evaluation_splits",
        "n_skipped_splits",
    ]
    summary_path = output_dir / "indirect_recall_summary.csv"
    best_path = output_dir / "indirect_recall_best_layers.csv"
    write_csv(summary_path, summary_rows, fieldnames)
    write_csv(best_path, best_rows, fieldnames)
    print(f"\nSaved layer summary: {summary_path}")
    print(f"Saved best layers:   {best_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run grouped indirect-recall probes on functional-group activations"
    )
    parser.add_argument(
        "--config",
        "-c",
        default="config_indirect_fg_one_shot_8B.yaml",
        help="Path to the indirect-recall YAML configuration",
    )
    parser.add_argument(
        "--experiments",
        nargs="*",
        help="Run only the named enabled experiments",
    )
    parser.add_argument("--dataset", help="Override paths.dataset_file")
    parser.add_argument(
        "--extraction-config", help="Override paths.extraction_config"
    )
    parser.add_argument(
        "--activation-base-dir", help="Override paths.activation_base_dir"
    )
    parser.add_argument("--output-dir", help="Override paths.output_dir")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate configuration and expected paths without training probes",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        exit_code = run(Path(args.config), args)
    except (FileNotFoundError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()

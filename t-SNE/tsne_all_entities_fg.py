import os
import re
import csv
import yaml
import torch
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend: required for headless sbatch jobs (no display)
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

# ============================
# User Configuration Variables
# ============================

# Paths
CONFIG_YAML_PATH = '../config_extract_activation_fg_8B.yaml'
CSV_PATH = '../functional_group_dataset_2.csv'
ACTIVATIONS_BASE_DIR = '../activation_datasets_functional_groups_2/meta-llama-Llama-3.1-8B/'
OUTPUT_DIR = '../Results/results_tsne_plots_fg_all_8B_feature_fgs_pka'
SUMMARY_CSV_PATH = os.path.join(OUTPUT_DIR, 'silhouette_summary.csv')

ID_COLUMN = 'iupac_name'
NUM_SYMBOLS = 92                # total number of molecules in functional_group_dataset.csv
SELECTED_LAYERS = None   # 8B has 32 layers (0-31): early / mid / late
                         # 70B has 80 layers (0-79) 
                         # If None, all layers are processed. Otherwise, specify a list of layer numbers to process (e.g., [0, 15, 31]).

# Both features are scored with silhouette (treated as categorical labels).
# carbon_count only has 4 distinct values (3,4,5,6), so factorizing it works
# the same way as for functional_group, even though it's still drawn with a
# continuous colorbar in the plots.
CATEGORICAL_FEATURE = 'functional_group'
CONTINUOUS_FEATURE = 'carbon_count'

POINT_SIZE = 22
COLOR_MAP = plt.cm.rainbow


# ============================
# Config loading
# ============================

def load_entity_configs(yaml_path):
    """
    Read entity_type / prompt_name / template count directly from the
    extraction config, so this never drifts out of sync with the actual
    YAML (instead of hardcoding a duplicate {entity_type: template_count}
    table in this script).
    """
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    entities = config['extraction']['entities']
    entity_configs = []
    for entry in entities:
        entity_configs.append({
            'entity_type': entry['entity_type'],
            'prompt_name': entry['prompt_name'],
            'activations_per_symbol': len(entry['templates']),
        })
    return entity_configs


# ============================
# Data loading
# ============================

def load_symbols_and_features(csv_path, id_column, num_symbols, activations_per_symbol, features):
    """
    Load the first num_symbols rows and the specified features from the CSV.
    Each row has multiple activations (one per template), so feature values
    are repeated accordingly to line up with the flattened activation order.
    """
    df = pd.read_csv(csv_path)
    symbols = df[id_column].head(num_symbols).tolist()

    symbols_repeated = []
    for symbol in symbols:
        symbols_repeated.extend([symbol] * activations_per_symbol)

    features_dict = {}
    for feature in features:
        if feature not in df.columns:
            raise ValueError(f"Feature '{feature}' not found in CSV columns.")
        feature_values = df[feature].head(num_symbols).tolist()
        repeated = []
        for value in feature_values:
            repeated.extend([value] * activations_per_symbol)
        features_dict[feature] = repeated

    expected_length = num_symbols * activations_per_symbol
    for feature, values in features_dict.items():
        if len(values) != expected_length:
            raise ValueError(f"Feature '{feature}' has {len(values)} values, expected {expected_length}.")

    return symbols_repeated, features_dict


def build_layer_pattern(entity_type, prompt_name):
    """
    Filename format from save_activations(): '{entity_type}.{aggregation}.{prompt_name}.layer_{layer_ix}.pt'
    aggregation is fixed as 'last' per the current config.
    """
    return re.compile(rf'{re.escape(entity_type)}\.last\.{re.escape(prompt_name)}\.layer_(\d+)\.pt')


def get_layer_files(directory, entity_type, prompt_name):
    pattern = build_layer_pattern(entity_type, prompt_name)
    files = []
    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            files.append((int(match.group(1)), filename))
    return [filename for _, filename in sorted(files)]


def load_activations(file_path, num_symbols, activations_per_symbol):
    """
    Load activations from a .pt file and convert to numpy array.
    map_location='cpu' is required because these tensors were saved on a
    GPU node during extraction, but this script runs on CPU only.
    """
    tensor = torch.load(file_path, map_location='cpu')
    if isinstance(tensor, torch.Tensor):
        activations = tensor.cpu().numpy()
    elif isinstance(tensor, dict) and 'activations' in tensor:
        activations = tensor['activations'].cpu().numpy()
    else:
        raise ValueError(f"Unsupported tensor type in file {file_path}")

    expected_shape = num_symbols * activations_per_symbol
    if activations.shape[0] != expected_shape:
        raise ValueError(f"Expected {expected_shape} activations, but got {activations.shape[0]} in {file_path}")
    return activations


# ============================
# Analysis
# ============================

def perform_pca(data, n_components=50):
    pca = PCA(n_components=n_components, random_state=42)
    return pca.fit_transform(data), pca


def perform_tsne(data, n_components=2, perplexity=30):
    tsne = TSNE(n_components=n_components, random_state=42, perplexity=perplexity, n_iter=1000)
    return tsne.fit_transform(data)


def compute_silhouette(tsne_data, values):
    """
    Treat `values` as categorical labels (via pd.factorize) and compute the
    silhouette score on the 2D t-SNE coordinates. Works for both string
    labels (functional_group) and small-integer labels (carbon_count).
    """
    try:
        labels_numeric = pd.factorize(values)[0]
        if len(set(labels_numeric)) > 1:
            return float(silhouette_score(tsne_data, labels_numeric))
    except Exception:
        pass
    return None


def assign_colors_to_categories(categories):
    unique_categories = sorted(list(set(categories)))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_categories)))
    category_to_color = {c: colors[i] for i, c in enumerate(unique_categories)}
    return [category_to_color[c] for c in categories], unique_categories


# ============================
# Plotting
# ============================

def plot_tsne(tsne_data, features_dict, metrics, title, output_path):
    """
    metrics: dict {feature_name: silhouette_score_or_None}, computed
    externally so both the categorical feature and the few-valued
    "continuous" feature (carbon_count) get a score shown, regardless of
    which colormap style is used to draw them.
    """
    num_features = len(features_dict)
    fig, axes = plt.subplots(1, num_features, figsize=(6 * num_features, 6), squeeze=False)

    for idx, (feature, values) in enumerate(features_dict.items()):
        ax = axes[0, idx]
        ax.set_aspect('equal', adjustable='box')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(feature, fontsize=12)
        is_categorical = isinstance(values[0], str) or isinstance(values[0], bool)

        if is_categorical:
            category_colors, unique_categories = assign_colors_to_categories(values)
            ax.scatter(tsne_data[:, 0], tsne_data[:, 1], s=POINT_SIZE, c=category_colors, marker='o', alpha=0.7)
            handles = [
                plt.Line2D([0], [0], marker='o', color='w', label=cat, markerfacecolor=c, markersize=10)
                for cat, c in zip(unique_categories, plt.cm.rainbow(np.linspace(0, 1, len(unique_categories))))
            ]
            ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, title="")
        else:
            values_array = np.array(values, dtype=float)
            scatter = ax.scatter(tsne_data[:, 0], tsne_data[:, 1], s=POINT_SIZE, c=values_array,
                                 cmap=COLOR_MAP, marker='o', alpha=0.7)
            cbar = fig.colorbar(scatter, ax=ax, orientation='horizontal', pad=0.1)
            cbar.set_label(feature, fontsize=12)

        metric_val = metrics.get(feature)
        metric_text = f"Silhouette Score: {metric_val:.2f}" if metric_val is not None else "Silhouette Score: N/A"
        ax.text(0.95, 0.05, metric_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.5))

    plt.suptitle(title, fontsize=14)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


# ============================
# Main
# ============================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    entity_configs = load_entity_configs(CONFIG_YAML_PATH)
    print(f"Loaded {len(entity_configs)} entity configs from {CONFIG_YAML_PATH}")

    # Perplexity must be smaller than the number of data points; with only
    # 44 molecules, scale it down for more stable t-SNE results.
    perplexity = min(30, max(5, NUM_SYMBOLS // 3))
    summary_rows = []

    for entity_cfg in entity_configs:
        entity_type = entity_cfg['entity_type']
        prompt_name = entity_cfg['prompt_name']
        activations_per_symbol = entity_cfg['activations_per_symbol']
        activations_dir = os.path.join(ACTIVATIONS_BASE_DIR, entity_type)

        print(f"\n=== Entity: {entity_type} ({activations_per_symbol} templates) ===")

        try:
            symbols, features_dict = load_symbols_and_features(
                CSV_PATH, ID_COLUMN, NUM_SYMBOLS, activations_per_symbol,
                [CATEGORICAL_FEATURE, CONTINUOUS_FEATURE]
            )
        except Exception as e:
            print(f"  Error loading CSV features for {entity_type}: {e}")
            continue

        if not os.path.isdir(activations_dir):
            print(f"  Activations dir not found: {activations_dir}, skipping.")
            continue

        pattern = build_layer_pattern(entity_type, prompt_name)
        layer_files = get_layer_files(activations_dir, entity_type, prompt_name)
        if SELECTED_LAYERS is not None:
            layer_files = [f for f in layer_files if int(pattern.match(f).group(1)) in SELECTED_LAYERS]
        if not layer_files:
            print(f"  No matching layer files for {entity_type}, skipping.")
            continue

        for filename in layer_files:
            match = pattern.match(filename)
            original_layer_num = int(match.group(1))
            layer_num = original_layer_num + 1
            file_path = os.path.join(activations_dir, filename)
            print(f"  Processing {filename} (Layer {layer_num})")

            try:
                activations = load_activations(file_path, NUM_SYMBOLS, activations_per_symbol)
                pca_result, _ = perform_pca(activations, n_components=min(50, activations.shape[1]))
                tsne_result = perform_tsne(pca_result, perplexity=perplexity)
            except Exception as e:
                print(f"    Error processing layer {layer_num}: {e}")
                continue

            metrics = {
                CATEGORICAL_FEATURE: compute_silhouette(tsne_result, features_dict[CATEGORICAL_FEATURE]),
                CONTINUOUS_FEATURE: compute_silhouette(tsne_result, features_dict[CONTINUOUS_FEATURE]),
            }

            summary_rows.append({
                'entity_type': entity_type,
                'layer': layer_num,
                'silhouette_functional_group': metrics[CATEGORICAL_FEATURE],
                'silhouette_carbon_count': metrics[CONTINUOUS_FEATURE],
            })

            title = f"t-SNE — {entity_type}, Layer {layer_num} (8B)"
            output_path = os.path.join(OUTPUT_DIR, f"{entity_type}_layer_{layer_num}_tsne.png")
            try:
                plot_tsne(tsne_result, features_dict, metrics, title, output_path)
                print(f"    Saved plot to {output_path}")
            except Exception as e:
                print(f"    Error plotting layer {layer_num}: {e}")

    if not summary_rows:
        print("\nNo results to summarize.")
        return

    with open(SUMMARY_CSV_PATH, 'w', newline='') as f:
        fieldnames = ['entity_type', 'layer', 'silhouette_functional_group', 'silhouette_carbon_count']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nSaved summary table to {SUMMARY_CSV_PATH}")

    print("\n=== Summary (sorted by carbon_count silhouette, descending) ===")
    def sort_key(row):
        val = row['silhouette_carbon_count']
        return (val is None, -(val if val is not None else -999))
    for row in sorted(summary_rows, key=sort_key):
        fg = f"{row['silhouette_functional_group']:.2f}" if row['silhouette_functional_group'] is not None else "N/A"
        cc = f"{row['silhouette_carbon_count']:.2f}" if row['silhouette_carbon_count'] is not None else "N/A"
        print(f"  {row['entity_type']:<28} layer {row['layer']:>2}  functional_group={fg:>6}  carbon_count={cc:>6}")


if __name__ == "__main__":
    main()

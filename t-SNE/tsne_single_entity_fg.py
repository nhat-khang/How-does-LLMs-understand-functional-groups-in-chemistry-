import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import numpy as np
import re

# ============================
# User Configuration Variables
# ============================


# Paths
ACTIVATIONS_DIR = './activation_datasets_functional_groups_2/meta-llama-Llama-3.1-8B/functional_group/'
CSV_PATH = 'functional_group_dataset_2.csv'
OUTPUT_DIR = 'Results/results_tsne_plots_fg'

# Entity/template settings — MUST match config_extract_activation_fg_8B.yaml
ENTITY_TYPE = 'functional_group'      # entity being analyzed (matches .pt filename and folder name)
PROMPT_NAME = '10_templates'          # matches this entity's prompt_name in the config
ID_COLUMN = 'iupac_name'              # identifier column in the CSV (replaces 'Symbol')

# Visualization Settings
NUM_SYMBOLS = 92                       # total number of molecules in functional_group_dataset_2.csv
ACTIVATIONS_PER_SYMBOL = 10            # number of templates for this entity == number in PROMPT_NAME
# The 2 features match the 2 research questions in "Design Notes":
#  - functional_group: do molecules cluster by functional group (categorical)?
#  - carbon_count: within each cluster, is there a continuous direction by carbon count?
FEATURES_TO_USE = ['functional_group', 'carbon_count']
SELECTED_LAYERS = [0, 16, 31]          # 8B has 32 layers (0-31): early / mid / late

# Plot Settings
POINT_SIZE = 22
ANNOTATE_FONT_SIZE = 22
COLOR_MAP = plt.cm.rainbow
FPS_GIF = 2

# ============================
# Function Definitions
# ============================

def load_symbols_and_features(csv_path, id_column, num_symbols, activations_per_symbol, features):
    """
    Load the first num_symbols rows and the specified features from the CSV file.
    Each row has multiple activations (one per template), so features are repeated accordingly.
    id_column: name of the column used as the identifier for each row (replaces 'Symbol' from the original script).
    """
    df = pd.read_csv(csv_path)
    symbols = df[id_column].head(num_symbols).tolist()

    symbols_repeated = []
    for symbol in symbols:
        symbols_repeated.extend([symbol] * activations_per_symbol)

    features_dict = {}
    for feature in features:
        if feature is not None:
            if feature not in df.columns:
                raise ValueError(f"Feature '{feature}' not found in CSV columns.")
            feature_values = df[feature].head(num_symbols).tolist()
            features_dict[feature] = []
            for value in feature_values:
                features_dict[feature].extend([value] * activations_per_symbol)
        else:
            features_dict[feature] = [None] * (num_symbols * activations_per_symbol)

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
    """
    Get a sorted list of layer files from the specified directory.
    Sorting is based on the layer number extracted from the filename.
    """
    pattern = build_layer_pattern(entity_type, prompt_name)
    files = []
    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            layer_num = int(match.group(1))
            files.append((layer_num, filename))
    sorted_files = sorted(files, key=lambda x: x[0])
    return [filename for _, filename in sorted_files]


def load_activations(file_path, num_symbols, activations_per_symbol):
    """
    Load activations from a .pt file and convert to numpy array.
    """
    tensor = torch.load(file_path)
    if isinstance(tensor, torch.Tensor):
        activations = tensor.cpu().numpy()
    elif isinstance(tensor, dict):
        if 'activations' in tensor:
            activations = tensor['activations'].cpu().numpy()
        else:
            raise KeyError("Key 'activations' not found in the tensor dictionary.")
    else:
        raise ValueError(f"Unsupported tensor type in file {file_path}")

    expected_shape = num_symbols * activations_per_symbol
    if activations.shape[0] != expected_shape:
        raise ValueError(f"Expected {expected_shape} activations, but got {activations.shape[0]} in {file_path}")

    return activations


def perform_pca(data, n_components=50):
    pca = PCA(n_components=n_components, random_state=42)
    pca_result = pca.fit_transform(data)
    return pca_result, pca


def perform_tsne(data, n_components=2, perplexity=30):
    tsne = TSNE(n_components=n_components, random_state=42, perplexity=perplexity, n_iter=1000)
    tsne_result = tsne.fit_transform(data)
    return tsne_result


def assign_colors_to_categories(categories):
    unique_categories = sorted(list(set(categories)))
    num_categories = len(unique_categories)
    colors = plt.cm.rainbow(np.linspace(0, 1, num_categories))
    category_to_color = {category: colors[i] for i, category in enumerate(unique_categories)}
    category_colors = [category_to_color.get(cat, (0.5, 0.5, 0.5, 1.0)) for cat in categories]
    return category_colors, unique_categories


def plot_tsne_with_metrics(tsne_data, pca_data, labels, features_dict, title, output_path, annotate=False):
    num_features = len(features_dict)
    fig, axes = plt.subplots(1, num_features, figsize=(6 * num_features, 6), squeeze=False)

    for idx, (feature, values) in enumerate(features_dict.items()):
        ax = axes[0, idx]
        ax.set_aspect('equal', adjustable='box')
        ax.set_xticks([])
        ax.set_yticks([])
        is_categorical = isinstance(values[0], str) or isinstance(values[0], bool)
        metric_text = ""

        if is_categorical:
            category_colors, unique_categories = assign_colors_to_categories(values)
            ax.scatter(tsne_data[:, 0], tsne_data[:, 1],
                       s=POINT_SIZE, c=category_colors, marker='o', alpha=0.7)

            handles = [
                plt.Line2D([0], [0], marker='o', color='w', label=cat,
                           markerfacecolor=c, markersize=10)
                for cat, c in zip(unique_categories, plt.cm.rainbow(np.linspace(0, 1, len(unique_categories))))
            ]
            ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, title="")

            try:
                labels_numeric = pd.factorize(values)[0]
                if len(set(labels_numeric)) > 1:
                    silhouette_avg = silhouette_score(tsne_data, labels_numeric)
                    metric_text = f"Silhouette Score: {silhouette_avg:.2f}"
                else:
                    metric_text = "Silhouette Score: N/A"
            except Exception:
                metric_text = "Silhouette Score: N/A"

        elif feature is not None:
            values_array = np.array(values, dtype=float)
            mask = ~pd.isnull(values_array)
            num_present = np.sum(mask)

            if num_present < 2:
                ax.scatter(tsne_data[:, 0], tsne_data[:, 1],
                           s=POINT_SIZE, c='gray', marker='o', alpha=0.7)
            else:
                scatter_present = ax.scatter(tsne_data[mask, 0], tsne_data[mask, 1],
                                             s=POINT_SIZE, c=values_array[mask], cmap=COLOR_MAP,
                                             norm=plt.Normalize(np.nanmin(values_array[mask]), np.nanmax(values_array[mask])),
                                             marker='o', alpha=0.7)
                ax.scatter(tsne_data[~mask, 0], tsne_data[~mask, 1],
                           s=POINT_SIZE, c='gray', marker='x', alpha=0.7)

                cbar = fig.colorbar(scatter_present, ax=ax, orientation='horizontal', pad=0.1)
                cbar.set_label(feature, fontsize=14)

        else:
            ax.scatter(tsne_data[:, 0], tsne_data[:, 1],
                       s=POINT_SIZE, c='gray', marker='o', alpha=0.7)

        if metric_text:
            ax.text(0.95, 0.05, metric_text, transform=ax.transAxes,
                    fontsize=12, verticalalignment='bottom', horizontalalignment='right',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.5))

    plt.suptitle(title, fontsize=14)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


# ============================
# Main Function
# ============================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        symbols, features_dict = load_symbols_and_features(
            CSV_PATH,
            id_column=ID_COLUMN,
            num_symbols=NUM_SYMBOLS,
            activations_per_symbol=ACTIVATIONS_PER_SYMBOL,
            features=FEATURES_TO_USE
        )
    except ValueError as e:
        print(f"Error loading symbols and features: {e}")
        return

    layer_files = get_layer_files(ACTIVATIONS_DIR, ENTITY_TYPE, PROMPT_NAME)
    if not layer_files:
        print(f"No layer files found in {ACTIVATIONS_DIR} for entity '{ENTITY_TYPE}'.")
        return

    pattern = build_layer_pattern(ENTITY_TYPE, PROMPT_NAME)

    if SELECTED_LAYERS is not None:
        layer_files = [f for f in layer_files if int(pattern.match(f).group(1)) in SELECTED_LAYERS]
        if not layer_files:
            print("No matching layer files found for the selected layers.")
            return

    total_layers = len(layer_files)
    annotate_threshold = int(total_layers * 0.5)

    # Perplexity must be smaller than the number of data points; with only 44
    # molecules, lower it for more stable t-SNE results instead of keeping 30 (which was calibrated for 50+ points before).
    perplexity = min(30, max(5, NUM_SYMBOLS // 3))

    for idx, filename in enumerate(layer_files):
        match = pattern.match(filename)
        if not match:
            print(f"Filename {filename} does not match the expected pattern.")
            continue
        original_layer_num = int(match.group(1))
        layer_num = original_layer_num + 1
        file_path = os.path.join(ACTIVATIONS_DIR, filename)
        print(f"Processing {filename} (Layer {layer_num})")

        try:
            activations = load_activations(file_path, num_symbols=NUM_SYMBOLS, activations_per_symbol=ACTIVATIONS_PER_SYMBOL)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue

        try:
            pca_result, pca = perform_pca(activations, n_components=min(50, activations.shape[1]))
            explained_variance = pca.explained_variance_ratio_.sum()
            print(f"PCA explained variance ratio for Layer {layer_num}: {explained_variance:.2f}")
        except Exception as e:
            print(f"Error performing PCA on Layer {layer_num}: {e}")
            continue

        try:
            tsne_result = perform_tsne(pca_result, n_components=2, perplexity=perplexity)
        except Exception as e:
            print(f"Error performing t-SNE on Layer {layer_num}: {e}")
            continue

        annotate = idx >= annotate_threshold
        title = f"t-SNE — Layer {layer_num} ({ENTITY_TYPE}, 8B)"
        output_path = os.path.join(OUTPUT_DIR, f"layer_{layer_num}_tsne_{ENTITY_TYPE}.png")
        try:
            plot_tsne_with_metrics(
                tsne_data=tsne_result,
                pca_data=pca_result,
                labels=symbols,
                features_dict=features_dict,
                title=title,
                output_path=output_path,
                annotate=annotate
            )
            print(f"Saved t-SNE plot to {output_path}")
        except Exception as e:
            print(f"Error plotting t-SNE for Layer {layer_num}: {e}")


if __name__ == "__main__":
    main()

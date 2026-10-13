# How-does-LLMs-understand-functional-groups-in-chemistry-

We want to investigate how functional groups are represented internally within LLMs. This repository contains the code, files and results we have obtained during our research.

## Research question

Does Llama-3.1-8B really "understand" organic chemistry (functional groups), or does it only use shortcuts in the text, such as the endings of IUPAC names (`-ol`, `-one`) or the formula written in the prompt?

We look inside the model (its hidden activations) and ask:

1. Is the information about functional groups and molecular properties **there**?
2. **Where** (which layer) does it appear, and how is it organized?
3. When does the model get **ready to say** the answer?
4. Does what the model learned **transfer** to functional groups it has never seen?

## Dataset

| Item | Value |
|---|---|
| File | `functional_group_dataset_2.csv` |
| Molecules | 92 |
| Functional-group families | 20 (alcohol, amine, ketone, aldehyde, ester, amide, nitrile, thiol, thioether, sulfone, sulfoxide, alkyl halides, nitro, imine, alkane, ...) |
| Entities (what we ask about) | 11: `functional_group`, `functional_group_structure`, `mw`, `pka`, `pkah`, `tpsa`, `avg_carbon_oxidation_state`, `hbd`, `hba`, `boiling_point_c`, `water_solubility` |
| Prompts | 9–11 templates per entity |

Note: molecule structures are written in condensed notation. An earlier version mixed SMILES and condensed notation, which created a text-style confound; 48 rows were fixed.

## Model and settings

| Field | Value |
|---|---|
| Model | Llama-3.1-8B (32 layers, hidden size 4096) |
| Precision | float16, no quantization |
| Batch size | 50 |
| Hardware | 1 GPU (HCC Swan cluster, SLURM) |
| Activation used | Last token, every layer |

## Experiments

| # | Experiment | Question | Script / folder |
|---|---|---|---|
| 0 | **Extract activations** | Save the hidden states of the model for every prompt | `extract_activations.py`, `Data extractor/` |
| 1 | **t-SNE** | Do activations sort themselves by functional group? | `t-SNE/` |
| 2 | **Direct recall** | Can a linear probe read the label from the activation? | `Direct_recall/` |
| 3 | **Indirect recall** (molecule held out) | Does the label survive when the prompt asks about something else? | `Indirect_recall/`, `config_indirect_fg.yaml` |
| 4 | **Indirect recall** (functional group held out) | Does it work for a group never seen in training? | `Indirect_recall/`, `config_indirect_fg_modify_split.yaml` |
| 5 | **Position probe** | Is the position of the group (1 or 2) encoded? | `Position_test/`, `config_extract_fg_position_ablation_8B.yaml` |
| 6 | **Logit lens** | At which layer can the model's output layer "read" the answer? | `logit_lens/` |
| 7 | **Intervention** | If we patch in a class's average activation, does the model name that group? | `functional_group_intervention.py` (add to the repo when ready) |

### How each one works (short)

- **t-SNE** — No labels are used. PCA (50 dims) then t-SNE (2D). We colour points by functional group, boiling point, etc. and measure how tight the groups are with the silhouette score.
- **Direct recall** — Linear SVM/SVR probe for each of 11 entities × 32 layers. Cross-validation is **grouped by molecule**, so templates of one molecule never appear in both train and test.
- **Indirect recall** — Train on prompts about one property, predict a different label. Two splits: hold out molecules (easier) or hold out whole functional groups (strict).
- **Logit lens** — Apply the model's final norm and output layer to each layer's hidden state. We read the top-5 words per layer. A cell is TRUE when the right answer (or a valid chemical symbol such as `OH`, `NH`) is in the top-5.
- **Intervention** — Patch the average activation of one class into a neutral prompt and check which functional group the model names.

## Main findings so far

- **The information is present, and early.** Most labels (mw, pKa, pKaH, TPSA, H-bond counts) are readable by a probe at layer 0. This points to clues in the prompt (names, formulas) rather than deep chemistry.
- **Geometry changes around layer 17.** t-SNE shows a large jump in functional-group organization between layers 16 and 17, strongest for pKa prompts. Carbon count does not drive it.
- **pKa has two stages.** "Concept" words (`pH`, `acid`) show up first (about layers 17–24); the number only shows up at about layers 28–32.
- **Functional-group answers appear at different layers for different classes** (alcohol at layer 17, alkyl bromide and imine at 29–31). Some answers are built piece by piece (e.g. `merc` → mercapto).
- **Scoring matters.** Counting valid symbols (`OH`, `NH`, `SO`, ...) raises the best top-5 match rate from 56.5% to 79.3%.
- **Transfer to unseen groups is weak.** With whole functional groups held out: mw R² ≈ 0.49 (moderate), boiling point ≈ 0.16 (weak), pKa and pKaH fail.
- **Water solubility** shows no structure anywhere.

### Honest limits

- Top-5 matching is lenient; at the last layer the top-1 word is always `1` (a counting habit), so top-1 should be reported too.
- All molecules are textbook-common. We cannot yet tell real understanding from remembered textbook patterns.
- t-SNE and probes show what is **present**, not what the model **uses**. Causal evidence needs the intervention experiments.
- Part of the signal may come from the last-token position and prompt length.

## Repository layout

```
.
├── Data extractor/                              # code to build the activation data
├── Direct_recall/                               # experiment 2
├── Indirect_recall/                             # experiments 3 and 4
├── Position_test/                               # experiment 5
├── t-SNE/                                       # experiment 1
├── logit_lens/                                  # experiment 6
├── Results/                                     # CSVs, heatmaps, plots
├── activation_datasets_functional_groups_2/     # saved activations (main dataset)
├── activation_datasets_fg_position_ablation/    # saved activations (position test)
├── config_extract_activation_fg_8B.yaml         # extraction settings (main)
├── config_extract_fg_position_ablation_8B.yaml  # extraction settings (position test)
├── config_indirect_fg.yaml                      # indirect recall, molecule split
├── config_indirect_fg_modify_split.yaml         # indirect recall, functional-group split
├── extract_activations.py
├── functional_group_dataset_2.csv
└── README.md
```

The activation folders are large (see `.gitattributes` for how they are stored).

## How to run

The scripts were run on the HCC Swan cluster with SLURM and a conda environment named `chemical`. Paths inside the scripts are relative, so run everything **from the repository root**.

```bash
# 0. Extract activations (GPU job)
python extract_activations.py   # settings come from config_extract_activation_fg_8B.yaml

# 1. Direct recall, t-SNE, indirect recall, position test, logit lens:
#    run the script inside each folder (see the .sh / .slurm job file next to it)
sbatch <job_file>
```

GPU jobs need `--partition=gpu --gres=gpu:1` (and your account). Check a job with `squeue -u $USER` or `sacct -j <jobid>`.

## Outputs

| Output | Meaning |
|---|---|
| `direct_recall_summary.csv` | 11 entities × 32 layers, score per layer |
| `silhouette_summary.csv` | t-SNE silhouette per entity and layer |
| `logit_lens_*_summary_92mol*.csv` | top-5 words and TRUE/FALSE per molecule and layer |
| `*_heatmap_92mol*.png` | heatmap: molecule × layer, coloured when the answer is in the top-5 |
| `*_layer_N_tsne.png` | t-SNE map for one layer |

## Next steps

- Report top-1 match rate next to top-5.
- Analyze the intervention results.
- Look at the neighbours of each molecule on the activation map.
- Test molecules unlikely to appear in textbooks, to separate understanding from memory.
- Control: replace IUPAC names with formulas or anonymous IDs and see whether direct-recall scores collapse.
- Extend to the 70B model.

## Team

University of Nebraska–Lincoln. Research advisor: Dr. Janghoon. Researchers: Anh Nguyen, Tran, Khang.

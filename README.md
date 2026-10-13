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
| File | `functional_group_dataset.csv` |
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
| 1 | **t-SNE** | Do activations sort themselves by functional group? | `tsne_activation_all_layer.py` |
| 2 | **Direct recall** | Can a linear probe read the label from the activation? | `Direct_recall/direct_recall_fg.py` |
| 3 | **Indirect recall** (molecule held out) | Does the label survive when the prompt asks about something else? | `indirect_recall_fg.py` |
| 4 | **Indirect recall** (functional group held out) | Does it work for a group never seen in training? | `indirect_recall_fg.py` |
| 5 | **Position probe** | Is the position of the group (1 or 2) encoded? | — |
| 6 | **Logit lens** | At which layer can the model's output layer "read" the answer? | `logit_lens/` |
| 7 | **Intervention** | If we patch in a class's average activation, does the model name that group? | `functional_group_intervention.py` |

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

> Adjust to match your repo.

```
.
├── functional_group_dataset.csv
├── config_extract_activation_fg_8B.yaml     # activation extraction settings
├── Direct_recall/
│   └── direct_recall_fg.py
├── logit_lens/
│   ├── logit_lens_fg.py
│   ├── logit_lens_fg_all.py                 # 92 molecules, 5 entities
│   ├── logit_lens_fg_numeric.py
│   └── logit_lens_bare_name.py
├── tsne_activation_all_layer.py
├── indirect_recall_fg.py
├── functional_group_intervention.py
├── run_*.sh                                 # SLURM job scripts
└── Results/                                 # CSVs, heatmaps, t-SNE plots
```

## How to run

The scripts were run on the HCC Swan cluster with SLURM and a conda environment named `chemical`.

```bash
# 1. Go to the project folder (paths in the scripts are relative to it)
cd /path/to/Prof-Ock-data

# 2. Direct recall (CPU job)
sbatch run_direct_recall_fg_8B.sh

# 3. t-SNE (CPU job)
sbatch run_tsne_pka_mw.slurm

# 4. Logit lens (GPU job)
sbatch run_logit_lens_fg_all_8B.sh

# 5. Intervention (GPU job)
sbatch run_intervention_fg_8B.sh
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

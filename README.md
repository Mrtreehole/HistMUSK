# HistMUSK

HistMUSK predicts spot-level spatial gene expression from three aligned
histopathology modalities:

1. a spot image embedding (UNI in the reported experiments),
2. a variable-length set of CellViT cell embeddings, and
3. a pathology-language embedding.

The image branch anchors the representation. Cell embeddings are projected and
compressed into learned prototypes, queried by the image token, and added through
a bounded cell gate. The text representation is projected and concatenated with
the image-cell feature. A residual feed-forward block and expression decoder
produce one value per target gene.

## Repository contents

```text
HistMUSK/
|-- histmusk/          # model, data, training, metrics, and device code
|-- scripts/           # train, evaluate, predict, QC, and demo entry points
|-- configs/           # reported model and smoke-test configurations
|-- docs/              # input data contract
|-- tests/             # unit and integration tests
|-- pyproject.toml
`-- requirements.txt
```

Private datasets, model weights, target caches, and experiment outputs are not
included.

## Installation

Python 3.10 or newer is required. For GPU training, install the PyTorch build
matching the NVIDIA driver first, then install the project.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Verify the accelerator:

```bash
python scripts/diagnose_device.py --device auto
```

## End-to-end smoke run

This path creates synthetic, explicitly aligned multimodal data and exercises
target preparation, training, checkpointing, evaluation, and prediction.

```bash
python scripts/create_demo_data.py --output-root demo

python scripts/train.py \
  --config configs/smoke_test.yaml \
  --data-dir demo/multimodal \
  --expression-h5ad demo/expression.h5ad \
  --target-dir demo/targets \
  --output-dir demo/run \
  --device cpu

python scripts/evaluate.py \
  --data-dir demo/multimodal \
  --target-dir demo/targets \
  --checkpoint demo/run/checkpoints/best.pt \
  --split test \
  --output-dir demo/evaluation \
  --device cpu

python scripts/predict.py \
  --data-dir demo/multimodal \
  --target-dir demo/targets \
  --checkpoint demo/run/checkpoints/best.pt \
  --split test \
  --output-dir demo/predictions \
  --device cpu
```

## Real-data training

First validate the multimodal dataset and build the explicitly aligned target
cache:

```bash
python scripts/inspect_training_data.py \
  --data-dir /path/to/processed_multimodal_dataset \
  --expression-h5ad /path/to/gene_expression.h5ad \
  --target-dir /path/to/target_cache
```

Then train the reported 128-dimensional text model:

```bash
python scripts/train.py \
  --config configs/histmusk.yaml \
  --data-dir /path/to/processed_multimodal_dataset \
  --expression-h5ad /path/to/gene_expression.h5ad \
  --target-dir /path/to/target_cache \
  --qwen-h5ad /path/to/text_embeddings_128d.h5ad \
  --image-features-h5 /path/to/uni_features.h5 \
  --output-dir outputs/histmusk_seed42 \
  --device cuda:0
```

The model infers the image, cell, and gene dimensions from the prepared data.
The default config validates that the text width is 128. See
[`docs/data_format.md`](docs/data_format.md) for the complete contract.

## Independent ablations

Each command below starts from random initialization; no checkpoint is inherited.

```bash
# Image only
python scripts/train.py ... --disable-cell --disable-text

# Image + cells
python scripts/train.py ... --disable-text

# Image + text
python scripts/train.py ... --disable-cell

# Full HistMUSK
python scripts/train.py ...
```

Use a different output directory for every run. Evaluation additionally reports
inference-time `no_cell`, `no_text`, `spot_only`, and within-slide shuffled
controls, but these controls are not substitutes for independently trained
ablations.

## Leave-one-patient-out training

Use `run_patient_lopo.py` for strict patient-level cross-validation. In each
outer fold, one patient is held out for testing, a different patient is used
for validation and checkpoint selection, and all remaining patients form the
training set. The target scaler is fitted from that fold's training spots only.
No patient occurs in more than one split within a fold.

```bash
python scripts/run_patient_lopo.py \
  --config configs/histmusk.yaml \
  --data-dir /path/to/processed_multimodal_dataset \
  --expression-h5ad /path/to/gene_expression.h5ad \
  --target-dir /path/to/target_cache \
  --qwen-h5ad /path/to/text_embeddings_128d.h5ad \
  --image-features-h5 /path/to/uni_features.h5 \
  --output-root outputs/histmusk_patient_lopo \
  --devices cuda:0,cuda:1,cuda:2,cuda:3
```

Run a subset with `--patients PATIENT_A PATIENT_B`. Each GPU handles one fold
at a time; folds are distributed round-robin across `--devices`. A completed
fold is skipped on rerun, while `--overwrite` recreates selected folds.

Each `held_out_*` directory contains the exact split table, fold-only target
scaler, training checkpoint, logs, test metrics, per-gene metrics, and explicit
spot-ID predictions. Root-level outputs include patient metrics,
`lopo_gene_pcc_by_patient.csv`, pooled out-of-fold gene metrics, and a protocol
manifest recording every test/validation patient assignment.

## Outputs

Training writes portable checkpoints, the resolved configuration, environment,
data manifests, training curves, and validation metrics. Evaluation writes JSON
and CSV summaries, per-gene Pearson/Spearman correlations, per-spot metrics,
per-slide metrics, and diagnostic plots.

Run the test suite with:

```bash
python -m pytest -q
```

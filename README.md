# MSAE Training

Training framework for Sparse Autoencoders on CLIP embeddings.

## Setup

```bash
pip install -e .
```

Requires: PyTorch, PyTorch Lightning, CLIP, wandb, scikit-learn, datasets.

## Project Structure

```
├── src/
│   ├── sae.py                        # SAE architectures (Autoencoder, MatryoshkaAutoencoder)
│   ├── train.py                      # Training loop (PyTorch Lightning)
│   ├── loss.py                       # Loss functions (reconstruction + sparsity)
│   ├── config.py                     # Dataclass configs for all SAE variants
│   ├── metrics.py                    # Training-time metrics
│   ├── callbacks.py                  # Training callbacks
│   ├── utils.py                      # Utilities (activation functions, etc.)
│   ├── precompute_activations.py     # Extract CLIP embeddings from datasets
│   ├── get_raw_data.py               # Download CC3M / ImageNet datasets
│   ├── generate_labels.py            # Generate classification labels
│   ├── generate_image_paths.py       # Extract image file paths
│   ├── generate_celeba_attributes.py # Extract CelebA attribute CSVs
│   └── upload_embeddings.py          # Upload embeddings to HuggingFace Hub
├── tests/
│   └── test_parity.py                # SAE forward-pass parity tests
├── vocab/
│   └── clip_disect_20k.txt           # CLIP-Dissect 20k vocabulary
└── pyproject.toml
```

## Reproducing the Experiments

### Step 1: Data Preparation

Download CC3M and ImageNet, then extract CLIP embeddings:

```bash
# Download datasets
python src/get_raw_data.py --dataset cc3m --output_dir /path/to/data

# Extract CLIP embeddings (ViT-L/14)
python src/precompute_activations.py \
    --backbone ViT-L/14 \
    --dataset cc3m \
    --split train \
    --modality image \
    --output_dir /path/to/embeddings

# Repeat for text modality
python src/precompute_activations.py \
    --backbone ViT-L/14 \
    --dataset cc3m \
    --split train \
    --modality text \
    --output_dir /path/to/embeddings
```

Generate labels and paths for evaluation datasets:

```bash
python src/generate_labels.py --backbone ViT-L/14 --output_dir /path/to/embeddings
python src/generate_image_paths.py --backbone ViT-L/14 --output_dir /path/to/embeddings
python src/generate_celeba_attributes.py --output_dir /path/to/embeddings
```

### Step 2: Train SAE Models

All SAE variants from the paper (8x expansion on ViT-L/14):

```bash
# ReLU SAEs
for LAMBDA in 0.03 0.003 0.001; do
    python src/train.py --config_name ReLUSAEConfig \
        --activation "ReLU" --l1_weight $LAMBDA \
        --expansion_factor 8 --backbone ViT-L~14 \
        --data_path /path/to/cc3m_ViT-L~14_train_image_2905954_768.npy
done

# TopK SAEs
for K in 32 64 128 256 512; do
    python src/train.py --config_name TopKSAEConfig \
        --activation "TopKReLU_${K}" \
        --expansion_factor 8 --backbone ViT-L~14 \
        --data_path /path/to/cc3m_ViT-L~14_train_image_2905954_768.npy
done

# BatchTopK SAEs
for K in 16 32 64 128 256; do
    python src/train.py --config_name BatchTopKSAEConfig \
        --activation "BatchTopKReLU_${K}" \
        --expansion_factor 8 --backbone ViT-L~14 \
        --data_path /path/to/cc3m_ViT-L~14_train_image_2905954_768.npy
done

# Matryoshka SAEs
python src/train.py --config_name MSAE_UWConfig \
    --expansion_factor 8 --backbone ViT-L~14 \
    --data_path /path/to/cc3m_ViT-L~14_train_image_2905954_768.npy

python src/train.py --config_name MSAE_RWConfig \
    --expansion_factor 8 --backbone ViT-L~14 \
    --data_path /path/to/cc3m_ViT-L~14_train_image_2905954_768.npy

# Matryoshka + L1 ablation
python src/train.py --config_name MSAE_UW_L1Config \
    --expansion_factor 8 --backbone ViT-L~14 \
    --l1_granularity_threshold 512 \
    --data_path /path/to/cc3m_ViT-L~14_train_image_2905954_768.npy

python src/train.py --config_name MSAE_RW_L1Config \
    --expansion_factor 8 --backbone ViT-L~14 \
    --l1_granularity_threshold 512 \
    --data_path /path/to/cc3m_ViT-L~14_train_image_2905954_768.npy
```

Repeat with `--expansion_factor 16` and `--expansion_factor 32` for all expansion rates.
Repeat with `--backbone ViT-B~16` for the second backbone.

### Step 3: Train Text-Modality Models

```bash
python src/train.py --config_name MSAE_UWConfig \
    --expansion_factor 8 --backbone ViT-L~14 \
    --data_path /path/to/cc3m_ViT-L~14_train_text_2905954_768.npy

python src/train.py --config_name MSAE_RWConfig \
    --expansion_factor 8 --backbone ViT-L~14 \
    --data_path /path/to/cc3m_ViT-L~14_train_text_2905954_768.npy
```

## Configurations

Defined in `src/config.py`:

| Config | LR | Sparsity | Activation |
|--------|-----|----------|------------|
| `ReLUSAEConfig` | 5e-5 | L1 (λ=0.003) | ReLU |
| `TopKSAEConfig` | 5e-4 | None (TopK enforced) | TopKReLU_k |
| `BatchTopKSAEConfig` | 5e-4 | None (BatchTopK enforced) | BatchTopKReLU_k |
| `MSAE_UWConfig` | 1e-4 | None | TopKReLU (nested, uniform weights) |
| `MSAE_RWConfig` | 1e-4 | None | TopKReLU (nested, reverse weights) |
| `MSAE_UW_L1Config` | 1e-4 | L1 (λ=0.003) | TopKReLU (nested + L1) |
| `MSAE_RW_L1Config` | 1e-4 | L1 (λ=0.003) | TopKReLU (nested + L1) |

## Data Format

Embeddings are stored as memory-mapped `.npy` files with naming convention:

```
{dataset}_{backbone}_{split}_{modality}_{n_samples}_{dim}.npy
```

Example: `cc3m_ViT-L~14_train_image_2905954_768.npy`

## Hardware

Training was performed on NVIDIA H100 GPUs. A single GPU with 80GB memory is sufficient for all expansion rates.

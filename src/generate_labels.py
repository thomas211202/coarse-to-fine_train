"""
Generate integer class-label .npy files for ImageNet train and validation splits.

The labels are extracted from the HuggingFace ILSVRC/imagenet-1k dataset in the
same iteration order used by precompute_activations.py, so they align index-for-index
with the existing embedding .npy files.

Usage:
    python src/generate_labels.py --split train
    python src/generate_labels.py --split validation

Output (written to $CACHE_DIR/outputs/embeddings/data/):
    imagenet_ViT-L~14_train_labels_1281167.npy
    imagenet_ViT-L~14_validation_labels_50000.npy
"""

import os
import argparse
import numpy as np
from tqdm import tqdm
from datasets import load_dataset

from utils import set_seed


SPLIT_SIZES = {
    "train": 1281167,
    "validation": 50000,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "validation"], required=True)
    return parser.parse_args()


def main(args):
    set_seed(42)

    cache_root = os.environ.get("CACHE_DIR")
    if cache_root is None:
        raise RuntimeError("CACHE_DIR must be set")

    os.environ["HF_HOME"] = os.path.join(cache_root, "hf")
    os.environ["HF_DATASETS_CACHE"] = os.path.join(cache_root, "hf", "datasets")
    os.environ["HF_HUB_CACHE"] = os.path.join(cache_root, "hf", "hub")

    out_dir = os.path.join(cache_root, "outputs", "embeddings", "data")
    os.makedirs(out_dir, exist_ok=True)

    n = SPLIT_SIZES[args.split]
    out_path = os.path.join(out_dir, f"imagenet_ViT-L~14_{args.split}_labels_{n}.npy")

    print(f"Loading ILSVRC/imagenet-1k {args.split} split ({n} samples)...")
    dataset = load_dataset("ILSVRC/imagenet-1k", split=args.split, streaming=False)

    labels = np.empty(n, dtype=np.int32)
    for i, sample in enumerate(tqdm(dataset, total=n, desc="Collecting labels")):
        labels[i] = sample["label"]

    np.save(out_path, labels)
    print(f"Saved {n} labels → {out_path}")


if __name__ == "__main__":
    main(parse_args())

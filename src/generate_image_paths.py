"""
Generate image-paths .npy files for ImageNet and CelebA splits.

get_raw_data.py saves images as {idx:08d}.jpg under:
    $CACHE_DIR/outputs/raw_datasets/{dataset}/{split}/images/

This script scans the actual files on disk (sorted) and saves their absolute
paths as a numpy array of strings. The count of discovered files is used to
name the output file, matching the embedding .npy filename convention.

Usage:
    python src/generate_image_paths.py --dataset imagenet --split validation
    python src/generate_image_paths.py --dataset imagenet --split train
    python src/generate_image_paths.py --dataset celeba   --split train

Output (written to $CACHE_DIR/outputs/embeddings/data/):
    imagenet_ViT-L~14_validation_image_paths_50000.npy
    imagenet_ViT-L~14_train_image_paths_1281167.npy
    celeba_ViT-L~14_train_image_paths_162770.npy
"""

import os
import argparse
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["imagenet", "celeba"], required=True)
    parser.add_argument("--split",   choices=["train", "validation"], required=True)
    return parser.parse_args()


def main(args):
    cache_root = os.environ.get("CACHE_DIR")
    if cache_root is None:
        raise RuntimeError("CACHE_DIR must be set")

    image_dir = os.path.join(
        cache_root, "outputs", "raw_datasets", args.dataset, args.split, "images"
    )

    if not os.path.isdir(image_dir):
        raise FileNotFoundError(
            f"Image directory not found: {image_dir}\n"
            f"Run get_raw_data.py first:  python src/get_raw_data.py -d {args.dataset}"
            + (" -s" if args.split == "train" else "")
        )

    # Scan actual files so the count matches reality even if some images failed
    files = sorted(f for f in os.listdir(image_dir) if f.endswith(".jpg"))
    if not files:
        raise FileNotFoundError(f"No .jpg files found in {image_dir}")

    paths = np.array(
        [os.path.join(image_dir, f) for f in files],
        dtype=object,
    )
    n = len(paths)
    print(f"Found {n} images in {image_dir}")

    out_dir = os.path.join(cache_root, "outputs", "embeddings", "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(
        out_dir, f"{args.dataset}_ViT-L~14_{args.split}_image_paths_{n}.npy"
    )

    np.save(out_path, paths)
    print(f"Saved {n} paths → {out_path}")


if __name__ == "__main__":
    main(parse_args())

"""
Convert the CelebA attribute file downloaded by torchvision into two CSV files
(train split and val split) containing all 40 attributes with a proper header row.

The torchvision CelebA class downloads list_attr_celeba.txt to:
    $CACHE_DIR/datasets/celeba/celeba/list_attr_celeba.txt

and list_eval_partition.txt to:
    $CACHE_DIR/datasets/celeba/celeba/list_eval_partition.txt

Usage:
    python src/generate_celeba_attributes.py

Output (written to $CACHE_DIR/outputs/embeddings/data/):
    celeba_train_attributes.csv    (162,770 rows)
    celeba_val_attributes.csv      (19,867 rows)
"""

import os
import csv
import numpy as np


ATTR_NAMES = [
    "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes",
    "Bald", "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair",
    "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin",
    "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones",
    "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard",
    "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline",
    "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair",
    "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace",
    "Wearing_Necktie", "Young",
]


def main():
    cache_root = os.environ.get("CACHE_DIR")
    if cache_root is None:
        raise RuntimeError("CACHE_DIR must be set")

    celeba_root = os.path.join(cache_root, "datasets", "celeba")
    attr_file = os.path.join(celeba_root, "list_attr_celeba.txt")
    partition_file = os.path.join(celeba_root, "list_eval_partition.txt")

    if not os.path.exists(attr_file):
        raise FileNotFoundError(
            f"CelebA attribute file not found: {attr_file}\n"
            "Run precompute_activations.py -d celeba first to trigger the download."
        )

    # Parse list_attr_celeba.txt
    # Format: first line = count, second line = attr names, remaining = filename + 40 values (-1/1)
    with open(attr_file, "r") as f:
        lines = f.readlines()
    n_total = int(lines[0].strip())
    data_lines = lines[2:]  # skip count and header

    # Parse partition file (0=train, 1=val, 2=test)
    partition = {}
    with open(partition_file, "r") as f:
        for line in f:
            fname, part = line.strip().split()
            partition[fname] = int(part)

    train_rows, val_rows = [], []
    for line in data_lines:
        parts = line.strip().split()
        fname = parts[0]
        # Convert -1/1 to 0/1
        attrs = [(int(x) + 1) // 2 for x in parts[1:]]
        row = dict(zip(ATTR_NAMES, attrs))
        row["filename"] = fname
        part = partition.get(fname, 0)
        if part == 0:
            train_rows.append(row)
        elif part == 1:
            val_rows.append(row)

    out_dir = os.path.join(cache_root, "outputs", "embeddings", "data")
    os.makedirs(out_dir, exist_ok=True)

    fieldnames = ["filename"] + ATTR_NAMES
    for name, rows in [("train", train_rows), ("val", val_rows)]:
        out_path = os.path.join(out_dir, f"celeba_{name}_attributes.csv")
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved {len(rows)} rows → {out_path}")


if __name__ == "__main__":
    main()

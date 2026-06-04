import os
import argparse
import logging
from tqdm import tqdm
from PIL import Image

import torch
from torch.utils.data import DataLoader

from utils import set_seed
from precompute_activations import (
    SUPPORTED_DATASETS,
    SUPPORTED_VOCABS,
    load_data,
    load_vocab,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser("Save dataset (JPEGs + text)")
    parser.add_argument("-d", "--dataset", type=str, required=True)
    parser.add_argument("-s", "--train-split", action="store_true")
    parser.add_argument("-b", "--batch-size", type=int, default=128)
    parser.add_argument("-w", "--workers", type=int, default=8)
    parser.add_argument("-v", "--vocab-size", type=int, default=-1)
    return parser.parse_args()


def raw_collate(batch):
    """Collate that keeps PIL images as a list (no torch.stack)."""
    images, texts = [], []
    for item in batch:
        if item is None:
            continue
        img, txt = item
        images.append(img)
        texts.append(txt)
    return images or None, texts or None


def save_image(img, path):
    if isinstance(img, torch.Tensor):
        img = img.permute(1, 2, 0).cpu().numpy()
        img = Image.fromarray(img.astype("uint8"))
    elif isinstance(img, Image.Image):
        img = img.convert("RGB")
    else:
        raise TypeError(f"Unsupported image type: {type(img)}")

    img.save(path, format="JPEG", quality=95, subsampling=0)


def main(args):
    set_seed(42)

    cache_root = os.environ.get("CACHE_DIR")
    if cache_root is None:
        raise RuntimeError("CACHE_DIR must be set")

    dataset_root = os.path.join(cache_root, "datasets")
    output_root = os.path.join(cache_root, "outputs", "raw_datasets")

    split = "train" if args.train_split else "validation"
    out_dir = os.path.join(output_root, args.dataset, split)
    image_dir = os.path.join(out_dir, "images")
    os.makedirs(image_dir, exist_ok=True)

    text_path = os.path.join(out_dir, "texts.txt")

    logger.info(f"Saving dataset={args.dataset}, split={split}")

    if args.dataset in SUPPORTED_DATASETS:
        dataset = load_data(
            args.dataset,
            preprocess=None,
            train=args.train_split,
            dataset_loc=dataset_root,
        )
    elif args.dataset in SUPPORTED_VOCABS:
        dataset = load_vocab(args.dataset, args.vocab_size)
    else:
        raise ValueError(f"Unsupported dataset {args.dataset}")

    dl = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=0,  # single-process: avoids OOM from workers buffering PIL images
        collate_fn=raw_collate,
        drop_last=False,
    )

    texts_out = []
    global_idx = 0

    logger.info("Writing JPEGs and text")
    for images, texts in tqdm(dl):
        batch_size = len(texts) if texts is not None else len(images)

        for i in range(batch_size):
            idx = global_idx + i

            if images is not None:
                img_path = os.path.join(image_dir, f"{idx:08d}.jpg")
                save_image(images[i], img_path)

            if texts is not None:
                texts_out.append(str(texts[i]))

        global_idx += batch_size

    if texts_out:
        with open(text_path, "w", encoding="utf-8") as f:
            f.write("\n".join(texts_out))

    logger.info(f"Saved {global_idx} samples to {out_dir}")
    logger.info("Dataset export complete.")


if __name__ == "__main__":
    args = parse_args()
    main(args)

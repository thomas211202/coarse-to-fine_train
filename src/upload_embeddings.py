import os
import re
import argparse
import logging
from huggingface_hub import HfApi

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Matches files produced by precompute_activations.py:
#   {dataset}_{model}_{split}_image_{N}_{D}.npy
#   {dataset}_{model}_{split}_text_{N}_{D}.npy
#   {dataset}_{model}_{split}_text_{N}.txt
_EMB_RE  = re.compile(r"^(.+)_(ViT-[^_]+|RN\d+)_(train|validation)_(image|text)_\d+_\d+\.npy$")
_TEXT_RE = re.compile(r"^(.+)_(ViT-[^_]+|RN\d+)_(train|validation)_text_\d+\.txt$")


def parse_args():
    parser = argparse.ArgumentParser("Upload precomputed embeddings to a HF dataset repo")
    parser.add_argument("--hf-repo", required=True,
                        help="HuggingFace dataset repo (e.g. 'username/msae-embeddings')")
    parser.add_argument("--embeddings-dir", default=None,
                        help="Path to embeddings/data dir. Defaults to $CACHE_DIR/outputs/embeddings/data")
    return parser.parse_args()


def main():
    args = parse_args()

    emb_dir = args.embeddings_dir
    if emb_dir is None:
        cache_root = os.environ.get("CACHE_DIR")
        if not cache_root:
            raise RuntimeError("Set --embeddings-dir or CACHE_DIR")
        emb_dir = os.path.join(cache_root, "outputs", "embeddings", "data")

    if not os.path.isdir(emb_dir):
        raise FileNotFoundError(f"Embeddings directory not found: {emb_dir}")

    logger.info(f"Scanning {emb_dir}")
    files = os.listdir(emb_dir)

    # Build map: (dataset, model, split) -> {image, text_emb, text_labels}
    slots: dict[tuple, dict] = {}

    for fname in files:
        m = _EMB_RE.match(fname)
        if m:
            dataset, model, split, kind = m.group(1), m.group(2), m.group(3), m.group(4)
            key = (dataset, model, split)
            slots.setdefault(key, {})
            slots[key]["image" if kind == "image" else "text_emb"] = os.path.join(emb_dir, fname)
            continue

        m = _TEXT_RE.match(fname)
        if m:
            dataset, model, split = m.group(1), m.group(2), m.group(3)
            key = (dataset, model, split)
            slots.setdefault(key, {})
            slots[key]["text_labels"] = os.path.join(emb_dir, fname)

    if not slots:
        logger.warning("No embedding files found in %s", emb_dir)
        return

    api = HfApi()

    for (dataset, model, split), paths in sorted(slots.items()):
        hf_dir = f"{dataset}/{model}/{split}"
        logger.info(f"Uploading {dataset}/{model}/{split} ({list(paths)})")

        for kind, repo_name in [("image", "image_embeddings.npy"),
                                 ("text_emb", "text_embeddings.npy"),
                                 ("text_labels", "labels.txt")]:
            local = paths.get(kind)
            if not local:
                logger.warning(f"  Missing {kind}, skipping")
                continue
            repo_path = f"{hf_dir}/{repo_name}"
            logger.info(f"  {os.path.basename(local)} → {repo_path}")
            api.upload_file(
                path_or_fileobj=local,
                path_in_repo=repo_path,
                repo_id=args.hf_repo,
                repo_type="dataset",
            )
            logger.info(f"  Done: {repo_path}")

    logger.info("All uploads complete.")


if __name__ == "__main__":
    main()

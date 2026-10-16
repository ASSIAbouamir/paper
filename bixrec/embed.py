"""Encode item text with a frozen sentence encoder.

    python -m bixrec.embed --data data/ml1m --model sentence-transformers/all-MiniLM-L6-v2

Writes ``<data>/text_emb.pt`` : float tensor (num_items + 1, dim), row 0 = zeros.
"""

import argparse
import os

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--batch", type=int, default=256)
    a = ap.parse_args()

    from sentence_transformers import SentenceTransformer

    ds = torch.load(os.path.join(a.data, "dataset.pt"), weights_only=False)
    texts = [t if t else "unknown item" for t in ds["item_text"][1:]]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    enc = SentenceTransformer(a.model, device=dev)
    emb = enc.encode(texts, batch_size=a.batch, convert_to_tensor=True,
                     normalize_embeddings=True, show_progress_bar=True).float().cpu()
    emb = torch.cat([torch.zeros(1, emb.shape[1]), emb], 0)
    torch.save({"emb": emb, "model": a.model}, os.path.join(a.data, "text_emb.pt"))
    print("saved", tuple(emb.shape))


if __name__ == "__main__":
    main()

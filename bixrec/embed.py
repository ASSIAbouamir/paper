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
    ap.add_argument("--encoder", choices=["sbert", "lsa"], default="sbert",
                    help="sbert: frozen sentence encoder (needs internet once); "
                         "lsa: TF-IDF + truncated SVD fitted on item text only (offline fallback)")
    ap.add_argument("--dim", type=int, default=128, help="LSA dimension")
    a = ap.parse_args()

    ds = torch.load(os.path.join(a.data, "dataset.pt"), weights_only=False)
    texts = [t if t else "unknown item" for t in ds["item_text"][1:]]
    if a.encoder == "lsa":
        import numpy as np
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        X = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b",sublinear_tf=True, min_df=1).fit_transform(texts)
        dim = min(a.dim, X.shape[1] - 1)
        Z = TruncatedSVD(dim, random_state=0).fit_transform(X)
        Z = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)
        emb = torch.cat([torch.zeros(1, dim), torch.tensor(Z, dtype=torch.float32)], 0)
        torch.save({"emb": emb, "model": f"tfidf-lsa-{dim}"}, os.path.join(a.data, "text_emb.pt"))
        print("saved", tuple(emb.shape), "vocab", X.shape[1])
        return

    from sentence_transformers import SentenceTransformer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    enc = SentenceTransformer(a.model, device=dev)
    emb = enc.encode(texts, batch_size=a.batch, convert_to_tensor=True,
                     normalize_embeddings=True, show_progress_bar=True).float().cpu()
    emb = torch.cat([torch.zeros(1, emb.shape[1]), emb], 0)
    torch.save({"emb": emb, "model": a.model}, os.path.join(a.data, "text_emb.pt"))
    print("saved", tuple(emb.shape))


if __name__ == "__main__":
    main()

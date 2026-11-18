"""Encode item text with a frozen encoder (fitted on item text only, never on interactions).

    python -m bixrec.embed --data data/ml1m                                  # MiniLM -> text_emb.pt
    python -m bixrec.embed --data data/sports --encoder lsa --name lsa        # -> text_emb_lsa.pt
    python -m bixrec.embed --data data/beauty_text --encoder sbert --whiten --name sbert_w

Writes ``<data>/text_emb.pt`` (or ``text_emb_<name>.pt``): float tensor (num_items + 1, dim), row 0 = zeros.
Encoders: sbert (frozen sentence encoder, needs internet once), lsa (TF-IDF + truncated SVD), tfidf (no SVD;
for short attribute vocabularies). --whiten applies PCA whitening (zero mean, identity covariance), which
WhitenRec (ICDE 2024) found important for anisotropic pre-trained text embeddings.
"""

import argparse
import os

import numpy as np
import torch


def whiten(z: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """PCA whitening of item vectors (rows); drops directions with ~zero variance."""
    z = z - z.mean(0, keepdims=True)
    cov = z.T @ z / len(z)
    val, vec = np.linalg.eigh(cov)
    keep = val > eps * val.max()
    return z @ vec[:, keep] / np.sqrt(val[keep])


def encode(texts, encoder: str, dim: int, model: str, batch: int) -> np.ndarray:
    if encoder in ("lsa", "tfidf"):
        from sklearn.feature_extraction.text import TfidfVectorizer
        x = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", sublinear_tf=True, min_df=1).fit_transform(texts)
        if encoder == "tfidf":
            return x.toarray().astype(np.float32)
        from sklearn.decomposition import TruncatedSVD
        z = TruncatedSVD(min(dim, x.shape[1] - 1), random_state=0).fit_transform(x)
        return z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
    from sentence_transformers import SentenceTransformer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    enc = SentenceTransformer(model, device=dev)
    return enc.encode(texts, batch_size=batch, normalize_embeddings=True, show_progress_bar=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--encoder", choices=["sbert", "lsa", "tfidf"], default="sbert")
    ap.add_argument("--dim", type=int, default=128, help="LSA dimension")
    ap.add_argument("--whiten", action="store_true")
    ap.add_argument("--name", default="", help="output text_emb_<name>.pt (default text_emb.pt)")
    a = ap.parse_args()

    ds = torch.load(os.path.join(a.data, "dataset.pt"), weights_only=False)
    texts = [t if t else "unknown item" for t in ds["item_text"][1:]]
    z = np.asarray(encode(texts, a.encoder, a.dim, a.model, a.batch), dtype=np.float64)
    if a.whiten:
        z = whiten(z)
    emb = torch.cat([torch.zeros(1, z.shape[1]), torch.tensor(z, dtype=torch.float32)], 0)
    desc = {"sbert": a.model, "lsa": f"tfidf-lsa-{z.shape[1]}", "tfidf": f"tfidf-{z.shape[1]}"}[a.encoder]
    out = os.path.join(a.data, f"text_emb_{a.name}.pt" if a.name else "text_emb.pt")
    torch.save({"emb": emb, "model": desc + ("+whiten" if a.whiten else "")}, out + ".tmp")
    os.replace(out + ".tmp", out)  # never leave a truncated file that later looks complete
    print("saved", out, tuple(emb.shape))


if __name__ == "__main__":
    main()

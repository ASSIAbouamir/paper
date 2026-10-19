"""
Generates 384D all-MiniLM-L6-v2 embeddings for KuaiRec items.
"""
import os
import sys
import pickle
import torch
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_PATH = os.path.join(BASE_DIR, "data", "kuairec", "processed_kuairec.pt")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "kuairec", "semantic_table_minilm.pt")

def main():
    print(f"[KuaiRec] Loading metadata from {PROCESSED_PATH}...")
    with open(PROCESSED_PATH, "rb") as f:
        data = pickle.load(f)

    item_metadata = data["item_metadata"]
    num_items = data["num_items"]
    print(f"[KuaiRec] Encoding {num_items} items with all-MiniLM-L6-v2...")

    texts = []
    for idx in range(1, num_items + 1):
        meta = item_metadata.get(idx, {})
        txt = meta.get("text", f"Video {idx}")
        texts.append(txt)

    encoder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    raw_embs = encoder.encode(texts, batch_size=128, show_progress_bar=True, convert_to_tensor=True).cpu()

    table = torch.zeros((num_items + 1, 384), dtype=torch.float32)
    table[1:] = raw_embs

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    torch.save(table, OUTPUT_PATH)
    print(f"[KuaiRec] Saved semantic table to {OUTPUT_PATH} (shape: {table.shape}).")

if __name__ == "__main__":
    main()

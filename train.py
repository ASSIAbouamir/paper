"""
Training Pipeline for SASRec Baseline and Proposed SASRec-LLM Hybrid Model.
Includes early stopping on validation NDCG@10, checkpoint saving, and differential learning rates.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import config
from data.preprocess import preprocess_movielens, load_data
from models.llm_encoder import extract_llm_embeddings
from utils.metrics import evaluate_batch_ranking, aggregate_metrics
from utils.diversity import calculate_ild

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from models.sasrec import SASRec
    from models.hybrid_model import SASRecLLMHybrid
    from utils.data_loader import get_dataloaders
    TORCH_AVAILABLE = True
except Exception as e:
    print(f"[Warning] PyTorch runtime unavailable in current environment ({e}). Training requires PyTorch + CUDA/CPU.")
    TORCH_AVAILABLE = False


def train_epoch(model, train_loader, optimizer, device):
    """Trains model for one epoch using Binary Cross-Entropy loss."""
    model.train()
    total_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()

    for log_seqs, pos_seqs, neg_seqs, _ in train_loader:
        log_seqs = log_seqs.to(device)
        pos_seqs = pos_seqs.to(device)
        neg_seqs = neg_seqs.to(device)

        optimizer.zero_grad()
        pos_logits, neg_logits = model(log_seqs, pos_seqs, neg_seqs)

        pos_masks = (pos_seqs != 0)
        pos_targets = torch.ones_like(pos_logits)
        neg_targets = torch.zeros_like(neg_logits)

        loss = criterion(pos_logits[pos_masks], pos_targets[pos_masks]) + \
               criterion(neg_logits[pos_masks], neg_targets[pos_masks])

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def evaluate_model(model, eval_loader, cold_start_users, device, k=10):
    """Evaluates model on validation or test dataset."""
    model.eval()
    metric_lists = []

    with torch.no_grad():
        for log_seqs, candidates, uids in eval_loader:
            log_seqs = log_seqs.to(device)
            candidates = candidates.to(device)

            scores = model.predict(log_seqs, candidates)
            batch_metrics = evaluate_batch_ranking(
                scores=scores,
                candidates=candidates,
                uids=uids,
                cold_start_users=cold_start_users,
                k=k
            )
            metric_lists.append(batch_metrics)

    return aggregate_metrics(metric_lists)


def train_model(model_type: str = "hybrid"):
    """
    Main training routine.
    model_type: "sasrec" or "hybrid"
    """
    if not TORCH_AVAILABLE:
        print(f"[Train] PyTorch is not loaded. Skipping training for '{model_type}'.")
        return None, {}

    if not os.path.exists(config.PROCESSED_DATA_PATH):
        processed_data = preprocess_movielens()
    else:
        processed_data = load_data(config.PROCESSED_DATA_PATH)

    if not os.path.exists(config.LLM_EMBEDDINGS_PATH):
        llm_data = extract_llm_embeddings(processed_data["item_metadata"])
    else:
        llm_data = load_data(config.LLM_EMBEDDINGS_PATH)

    device = torch.device(config.device if torch.cuda.is_available() and config.device == "cuda" else "cpu")
    print(f"\n==========================================")
    print(f" Starting Training: [{model_type.upper()}] on Device: {device}")
    print(f"==========================================")

    train_loader, val_loader, test_loader = get_dataloaders(processed_data, config)
    cold_start_users = processed_data["cold_start_users"]
    num_items = processed_data["num_items"]

    if model_type == "sasrec":
        model = SASRec(
            num_items=num_items,
            max_seq_len=config.max_seq_len,
            hidden_units=config.item_embed_dim,
            num_blocks=config.num_blocks,
            num_heads=config.num_heads,
            dropout_rate=config.dropout
        ).to(device)

        optimizer = optim.AdamW(model.parameters(), lr=config.lr_sasrec, weight_decay=config.weight_decay)

    elif model_type == "hybrid":
        llm_embs = llm_data["embeddings"].to(device) if isinstance(llm_data, dict) else llm_data.to(device)
        model = SASRecLLMHybrid(
            num_items=num_items,
            llm_embeddings=llm_embs,
            max_seq_len=config.max_seq_len,
            item_embed_dim=config.item_embed_dim,
            projected_llm_dim=config.projected_llm_dim,
            fusion_type=config.fusion_type,
            num_blocks=config.num_blocks,
            num_heads=config.num_heads,
            dropout_rate=config.dropout
        ).to(device)

        fusion_params = list(model.fusion_module.parameters())
        base_params = [p for n, p in model.named_parameters() if 'fusion_module' not in n and p.requires_grad]

        optimizer = optim.AdamW([
            {'params': base_params, 'lr': config.lr_sasrec},
            {'params': fusion_params, 'lr': config.lr_fusion}
        ], weight_decay=config.weight_decay)

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"best_{model_type}.pt")
    best_val_ndcg = 0.0
    patience_counter = 0

    for epoch in range(1, config.epochs + 1):
        start_time = time.time()
        loss = train_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate_model(model, val_loader, cold_start_users, device, k=config.top_k)
        elapsed = time.time() - start_time

        val_ndcg = val_metrics["NDCG@10"]
        val_hr = val_metrics["HR@10"]

        print(f"Epoch {epoch:02d}/{config.epochs:02d} [{elapsed:.1f}s] - Loss: {loss:.4f} | Val HR@10: {val_hr:.4f} | Val NDCG@10: {val_ndcg:.4f}")

        if val_ndcg > best_val_ndcg:
            best_val_ndcg = val_ndcg
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  --> Saved new best checkpoint to '{checkpoint_path}'")
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(f"[Early Stopping] Triggered at epoch {epoch}. Best Val NDCG@10: {best_val_ndcg:.4f}")
                break

    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
    test_metrics = evaluate_model(model, test_loader, cold_start_users, device, k=config.top_k)

    print(f"\n>>> Final [{model_type.upper()}] Test Results:")
    for m_name, m_val in test_metrics.items():
        print(f"    {m_name}: {m_val:.4f}")

    return model, test_metrics


if __name__ == "__main__":
    train_model("sasrec")
    train_model("hybrid")

"""
Training and Evaluation for Legal GNN Models.

Provides:
- train_outcome_model: Train for ONAMA/BOZMA prediction
- train_statute_model: Train for statute recommendation
- evaluate_model: Full evaluation with metrics
- run_experiment: Compare multiple models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
from collections import defaultdict
from typing import Dict, List, Tuple, Literal
from tqdm import tqdm
from pathlib import Path
import json

from .models import LegalGNN, create_model


def train_outcome_model(
    model: LegalGNN,
    data,
    num_epochs: int = 100,
    lr: float = 0.01,
    patience: int = 20,
    verbose: bool = True,
) -> Dict:
    """
    Train model for outcome prediction (ONAMA vs BOZMA).
    
    Args:
        model: LegalGNN model
        data: HeteroData with train/val/test masks
        num_epochs: Maximum epochs
        lr: Learning rate
        patience: Early stopping patience
        verbose: Print progress
        
    Returns:
        Dictionary with training history and final metrics
    """
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=10, factor=0.5)
    
    best_val_f1 = 0
    best_state = None
    patience_counter = 0
    history = {"loss": [], "val_f1": []}
    
    # Get labeled node indices - labeled_mask is boolean tensor for 111 labeled cases
    # The graph may have more case nodes (for statute recommendation) than labeled nodes
    n_total = data["case"].x.size(0)
    n_labeled = len(data["case"].y)
    
    # If we have more nodes than labels, the labels are for the FIRST n_labeled nodes
    # (this assumes labeled cases come first in the graph)
    if n_total > n_labeled:
        labeled_mask_full = torch.zeros(n_total, dtype=torch.bool)
        labeled_mask_full[:n_labeled] = True
    else:
        labeled_mask_full = torch.ones(n_total, dtype=torch.bool)
    
    pbar = tqdm(range(num_epochs), disable=not verbose, desc="Training outcome")
    
    for epoch in pbar:
        # Training
        model.train()
        optimizer.zero_grad()
        
        embeddings = model(data)
        logits = model.predict_outcome(embeddings)
        
        # Get logits only for labeled nodes (first n_labeled nodes)
        labeled_logits = logits[:n_labeled]
        
        train_mask = data["case"].train_mask
        loss = criterion(labeled_logits[train_mask], data["case"].y[train_mask])

        
        loss.backward()
        optimizer.step()
        history["loss"].append(loss.item())
        
        # Validation
        model.eval()
        with torch.no_grad():
            embeddings = model(data)
            logits = model.predict_outcome(embeddings)
            labeled_logits = logits[:n_labeled]
            
            val_mask = data["case"].val_mask
            val_pred = labeled_logits[val_mask].argmax(dim=1).numpy()
            val_true = data["case"].y[val_mask].numpy()
            val_f1 = f1_score(val_true, val_pred, average="weighted", zero_division=0)
        
        history["val_f1"].append(val_f1)
        scheduler.step(val_f1)
        
        # Early stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            if verbose:
                print(f"\nEarly stopping at epoch {epoch}")
            break
        
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "val_f1": f"{val_f1:.4f}"})
    
    # Restore best model
    if best_state:
        model.load_state_dict(best_state)
    
    # Test evaluation
    model.eval()
    with torch.no_grad():
        embeddings = model(data)
        logits = model.predict_outcome(embeddings)
        labeled_logits = logits[:n_labeled]
        
        test_mask = data["case"].test_mask
        test_pred = labeled_logits[test_mask].argmax(dim=1).numpy()
        test_true = data["case"].y[test_mask].numpy()
    
    return {
        "history": history,
        "accuracy": accuracy_score(test_true, test_pred),
        "f1": f1_score(test_true, test_pred, average="weighted", zero_division=0),
        "precision": precision_score(test_true, test_pred, average="weighted", zero_division=0),
        "recall": recall_score(test_true, test_pred, average="weighted", zero_division=0),
        "report": classification_report(test_true, test_pred, target_names=["BOZMA", "ONAMA"], zero_division=0),
        "confusion_matrix": confusion_matrix(test_true, test_pred).tolist(),
    }



def train_statute_model(
    model: LegalGNN,
    data,
    num_epochs: int = 100,
    lr: float = 0.01,
    verbose: bool = True,
) -> Dict:
    """
    Train model for statute recommendation (link prediction).
    
    Args:
        model: LegalGNN model
        data: HeteroData with edge splits
        num_epochs: Maximum epochs
        lr: Learning rate
        verbose: Print progress
        
    Returns:
        Dictionary with metrics
    """
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    
    edge_type = ("case", "cites", "statute")
    train_edges = data[edge_type].train_edge_index
    test_edges = data[edge_type].test_edge_index
    
    best_recall = 0
    best_state = None
    
    pbar = tqdm(range(num_epochs), disable=not verbose, desc="Training statute")
    
    for epoch in pbar:
        model.train()
        optimizer.zero_grad()
        
        embeddings = model(data)
        
        # Positive edges
        pos_score = model.predict_links(embeddings, train_edges)
        
        # Negative sampling
        n_edges = train_edges.size(1)
        neg_src = torch.randint(0, data["case"].x.size(0), (n_edges,))
        neg_dst = torch.randint(0, data["statute"].x.size(0), (n_edges,))
        neg_edges = torch.stack([neg_src, neg_dst])
        neg_score = model.predict_links(embeddings, neg_edges)
        
        # Loss
        scores = torch.cat([pos_score, neg_score])
        labels = torch.cat([torch.ones(n_edges), torch.zeros(n_edges)])
        loss = criterion(scores, labels)
        
        loss.backward()
        optimizer.step()
        
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        # Periodic evaluation
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                embeddings = model(data)
                all_scores = model.recommend_statutes(embeddings)
                recall = _compute_recall(all_scores, test_edges, k=10)
            
            if recall > best_recall:
                best_recall = recall
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    # Restore best
    if best_state:
        model.load_state_dict(best_state)
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        embeddings = model(data)
        all_scores = model.recommend_statutes(embeddings)
    
    return _evaluate_recommendation(all_scores, test_edges)


def _compute_recall(scores, test_edges, k=10):
    """Quick recall computation."""
    case_to_true = defaultdict(set)
    for i in range(test_edges.size(1)):
        case_to_true[test_edges[0, i].item()].add(test_edges[1, i].item())
    
    recalls = []
    for case_idx, true_set in case_to_true.items():
        top_k = set(scores[case_idx].topk(k).indices.tolist())
        recalls.append(len(top_k & true_set) / len(true_set))
    
    return np.mean(recalls) if recalls else 0


def _evaluate_recommendation(scores, test_edges, k_values=[5, 10, 20]):
    """Full recommendation metrics."""
    case_to_true = defaultdict(set)
    for i in range(test_edges.size(1)):
        case_to_true[test_edges[0, i].item()].add(test_edges[1, i].item())
    
    metrics = {f"precision@{k}": [] for k in k_values}
    metrics.update({f"recall@{k}": [] for k in k_values})
    metrics["mrr"] = []
    
    for case_idx, true_set in case_to_true.items():
        ranked = scores[case_idx].argsort(descending=True).tolist()
        
        for k in k_values:
            top_k = set(ranked[:k])
            hits = len(top_k & true_set)
            metrics[f"precision@{k}"].append(hits / k)
            metrics[f"recall@{k}"].append(hits / len(true_set))
        
        for rank, s in enumerate(ranked, 1):
            if s in true_set:
                metrics["mrr"].append(1.0 / rank)
                break
        else:
            metrics["mrr"].append(0.0)
    
    return {k: np.mean(v) for k, v in metrics.items()}


def run_experiment(
    data,
    gnn_types: List[str] = ["gat", "han", "hgt"],
    embedding_name: str = "unknown",
    num_epochs: int = 100,
    verbose: bool = True,
    save_checkpoints: bool = True,
) -> Dict:
    """
    Run complete experiment with multiple GNN types.
    
    Args:
        data: HeteroData object
        gnn_types: Models to compare
        embedding_name: Name of embedding model used
        num_epochs: Training epochs
        verbose: Print progress
        save_checkpoints: Save model checkpoints to checkpoints/ folder
        
    Returns:
        Results dictionary
    """
    results = {"embedding": embedding_name, "models": {}}
    
    # Create checkpoints directory
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    for gnn_type in gnn_types:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Training {gnn_type.upper()}")
            print("=" * 60)
        
        # Outcome prediction
        model = create_model(gnn_type, data)
        outcome = train_outcome_model(model, data, num_epochs=num_epochs, verbose=verbose)
        
        if verbose:
            print(f"Accuracy: {outcome['accuracy']:.4f}, F1: {outcome['f1']:.4f}")
        
        # Save outcome checkpoint
        if save_checkpoints:
            ckpt_path = checkpoint_dir / f"{gnn_type}_{embedding_name}_outcome.pt"
            torch.save({"state_dict": model.state_dict(), "meta": {"type": "outcome", "gnn": gnn_type}}, ckpt_path)
            if verbose:
                print(f"  Saved: {ckpt_path}")
        
        # Statute recommendation
        model_statute = create_model(gnn_type, data)
        statute = train_statute_model(model_statute, data, num_epochs=num_epochs, verbose=verbose)
        
        if verbose:
            print(f"Recall@10: {statute['recall@10']:.4f}, MRR: {statute['mrr']:.4f}")
        
        # Save statute checkpoint
        if save_checkpoints:
            ckpt_path = checkpoint_dir / f"{gnn_type}_{embedding_name}_statute.pt"
            torch.save({"state_dict": model_statute.state_dict(), "meta": {"type": "statute", "gnn": gnn_type}}, ckpt_path)
            if verbose:
                print(f"  Saved: {ckpt_path}")
        
        results["models"][gnn_type] = {
            "outcome": {k: v for k, v in outcome.items() if k != "history"},
            "statute": statute,
        }
    
    return results


def generate_report(results: Dict, output_file: str = None) -> str:
    """Generate markdown report."""
    lines = [
        "# GNN Experiment Results",
        f"\nEmbedding Model: **{results['embedding']}**",
        "\n## Outcome Prediction (ONAMA vs BOZMA)",
        "\n| Model | Accuracy | F1 | Precision | Recall |",
        "|-------|----------|-----|-----------|--------|",
    ]
    
    for model, res in results["models"].items():
        o = res["outcome"]
        lines.append(f"| {model.upper()} | {o['accuracy']:.4f} | {o['f1']:.4f} | {o['precision']:.4f} | {o['recall']:.4f} |")
    
    lines.extend([
        "\n## Statute Recommendation",
        "\n| Model | P@5 | R@5 | P@10 | R@10 | MRR |",
        "|-------|-----|-----|------|------|-----|",
    ])
    
    for model, res in results["models"].items():
        s = res["statute"]
        lines.append(f"| {model.upper()} | {s['precision@5']:.4f} | {s['recall@5']:.4f} | {s['precision@10']:.4f} | {s['recall@10']:.4f} | {s['mrr']:.4f} |")
    
    report = "\n".join(lines)
    
    if output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        Path(output_file).write_text(report, encoding="utf-8")
    
    return report

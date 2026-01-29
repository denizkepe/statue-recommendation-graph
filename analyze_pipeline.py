#!/usr/bin/env python3
"""
Complete Legal Graph Analysis Pipeline.

Runs parsing → embeddings → graph → training → analysis.
Outputs comprehensive statistics and visualizations.
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict
import sys

sys.path.insert(0, str(Path(__file__).parent))

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def print_header(title: str):
    """Print formatted header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def analyze_parsed_data(parsed_file: str) -> dict:
    """Analyze parsed data and print statistics."""
    print_header("PARSED DATA ANALYSIS")
    
    with open(parsed_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"\nTotal cases: {len(data)}")
    
    # Outcome distribution
    outcomes = Counter(c.get("outcome") for c in data)
    print("\nOutcome Distribution:")
    for o, c in outcomes.most_common():
        pct = 100 * c / len(data)
        bar = "█" * int(pct / 2)
        print(f"  {o or 'UNKNOWN':12} {c:4d} ({pct:5.1f}%) {bar}")
    
    # Chamber distribution
    chambers = Counter(c.get("chamber") for c in data if c.get("chamber"))
    print(f"\nChambers: {len(chambers)} unique")
    for ch, c in chambers.most_common(10):
        print(f"  {ch}: {c}")
    
    # Case type distribution (semi-dynamic field)
    case_types = Counter(c.get("case_type_enum") for c in data if c.get("case_type_enum"))
    if case_types:
        print(f"\nCase Types: {len(case_types)} unique")
        for ct, count in case_types.most_common():
            pct = 100 * count / len(data)
            print(f"  {ct}: {count} ({pct:.1f}%)")
    
    # Statute statistics
    all_statutes = []
    for c in data:
        all_statutes.extend(c.get("statute_ids", []))
    
    statute_counts = Counter(all_statutes)
    print(f"\nTotal statute citations: {len(all_statutes)}")
    print(f"Unique statutes: {len(statute_counts)}")
    print(f"Average statutes per case: {len(all_statutes) / len(data):.2f}")
    
    print("\nTop 15 Most Cited Statutes:")
    for s, c in statute_counts.most_common(15):
        print(f"  {s}: {c}")
    
    # Text length analysis
    text_lengths = [len(c.get("plaintiff_arguments", "")) for c in data]
    print(f"\nPlaintiff Arguments Text Length:")
    print(f"  Min: {min(text_lengths)}, Max: {max(text_lengths)}")
    print(f"  Mean: {np.mean(text_lengths):.0f}, Median: {np.median(text_lengths):.0f}")
    
    # Valid cases for graph - different criteria for different tasks
    # Outcome prediction: needs ONAMA/BOZMA labels
    outcome_valid = [c for c in data if c.get("outcome") in ["ONAMA", "BOZMA"] and c.get("statute_ids")]
    # Statute recommendation: just needs statutes
    statute_valid = [c for c in data if c.get("statute_ids")]
    
    print(f"\n✅ Valid cases for Outcome Prediction (ONAMA/BOZMA): {len(outcome_valid)} ({100*len(outcome_valid)/len(data):.1f}%)")
    print(f"✅ Valid cases for Statute Recommendation (any with statutes): {len(statute_valid)} ({100*len(statute_valid)/len(data):.1f}%)")
    
    return {
        "total": len(data),
        "valid_outcome": len(outcome_valid),
        "valid_statute": len(statute_valid),
        "outcomes": dict(outcomes),
        "n_statutes": len(statute_counts),
        "n_case_types": len(case_types),
        "top_statutes": statute_counts.most_common(20),
    }


def analyze_embeddings(embeddings_file: str, parsed_file: str) -> dict:
    """Analyze embeddings and print statistics."""
    print_header("EMBEDDING ANALYSIS")
    
    embeddings = torch.load(embeddings_file, weights_only=False)
    
    print(f"\nEmbedding shape: {embeddings.shape}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    
    emb_np = embeddings.numpy()
    
    print(f"\nStatistics:")
    print(f"  Mean: {emb_np.mean():.4f}")
    print(f"  Std: {emb_np.std():.4f}")
    print(f"  Min: {emb_np.min():.4f}")
    print(f"  Max: {emb_np.max():.4f}")
    
    # Norms
    norms = np.linalg.norm(emb_np, axis=1)
    print(f"\nL2 Norms:")
    print(f"  Mean: {norms.mean():.4f}")
    print(f"  Std: {norms.std():.4f}")
    
    # Pairwise similarities
    from sklearn.metrics.pairwise import cosine_similarity
    sims = cosine_similarity(emb_np)
    np.fill_diagonal(sims, 0)  # Exclude self-similarity
    
    print(f"\nPairwise Cosine Similarities:")
    print(f"  Mean: {sims.mean():.4f}")
    print(f"  Max: {sims.max():.4f}")
    print(f"  Min (non-self): {sims[sims > 0].min():.4f}")
    
    # High similarity pairs
    high_sim = (sims > 0.9).sum() // 2  # Divide by 2 for symmetry
    print(f"  Pairs with sim > 0.9: {high_sim}")
    
    return {
        "shape": embeddings.shape,
        "mean_sim": float(sims.mean()),
        "max_sim": float(sims.max()),
    }


def analyze_graph(graph_file: str) -> dict:
    """Analyze graph structure and print statistics."""
    print_header("GRAPH STRUCTURE ANALYSIS")
    
    data = torch.load(graph_file, weights_only=False)
    
    print(f"\nGraph Summary:")
    print(data)
    
    # Node statistics
    print("\n" + "-" * 40)
    print("NODE STATISTICS")
    print("-" * 40)
    
    for node_type in data.node_types:
        x = data[node_type].x
        print(f"\n{node_type.upper()}:")
        print(f"  Count: {x.size(0)}")
        print(f"  Feature dim: {x.size(1)}")
        print(f"  Feature mean: {x.mean():.4f}")
        print(f"  Feature std: {x.std():.4f}")
    
    # Edge statistics
    print("\n" + "-" * 40)
    print("EDGE STATISTICS")
    print("-" * 40)
    
    edge_stats = {}
    for edge_type in data.edge_types:
        edge_index = data[edge_type].edge_index
        n_edges = edge_index.size(1)
        src_type, rel, dst_type = edge_type
        
        print(f"\n{src_type} --[{rel}]--> {dst_type}:")
        print(f"  Edges: {n_edges}")
        
        # Degree statistics
        if n_edges > 0:
            src_degrees = torch.bincount(edge_index[0]).float()
            dst_degrees = torch.bincount(edge_index[1]).float()
            
            if len(src_degrees) > 0:
                print(f"  Src degree: mean={src_degrees.mean():.2f}, max={src_degrees.max():.0f}")
            if len(dst_degrees) > 0:
                print(f"  Dst degree: mean={dst_degrees.mean():.2f}, max={dst_degrees.max():.0f}")
        
        # Edge weights if available
        if hasattr(data[edge_type], 'edge_weight'):
            w = data[edge_type].edge_weight
            print(f"  Weight: mean={w.mean():.4f}, min={w.min():.4f}, max={w.max():.4f}")
        
        edge_stats[str(edge_type)] = n_edges
    
    # Label distribution
    if hasattr(data["case"], "y"):
        labels = data["case"].y
        print("\n" + "-" * 40)
        print("LABEL DISTRIBUTION")
        print("-" * 40)
        
        for i, name in enumerate(["BOZMA", "ONAMA"]):
            count = (labels == i).sum().item()
            pct = 100 * count / len(labels)
            print(f"  {name}: {count} ({pct:.1f}%)")
    
    # Train/Val/Test split
    if hasattr(data["case"], "train_mask"):
        print("\n" + "-" * 40)
        print("DATA SPLIT")
        print("-" * 40)
        print(f"  Train: {data['case'].train_mask.sum().item()}")
        print(f"  Val: {data['case'].val_mask.sum().item()}")
        print(f"  Test: {data['case'].test_mask.sum().item()}")
    
    return {
        "n_cases": data["case"].x.size(0),
        "n_statutes": data["statute"].x.size(0),
        "n_chambers": data["chamber"].x.size(0),
        "n_case_types": data["case_type"].x.size(0) if "case_type" in data.node_types else 0,
        "edge_stats": edge_stats,
    }


def visualize_graph(graph_file: str, output_dir: str):
    """Create graph visualizations."""
    print_header("GENERATING VISUALIZATIONS")
    
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    
    data = torch.load(graph_file, weights_only=False)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Node type distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    types = ['Cases', 'Statutes', 'Chambers']
    counts = [data["case"].x.size(0), data["statute"].x.size(0), data["chamber"].x.size(0)]
    bars = ax.bar(types, counts, color=['#2ecc71', '#3498db', '#9b59b6'])
    ax.set_ylabel('Count')
    ax.set_title('Node Types in Legal Graph')
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                str(count), ha='center', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path / 'node_types.png', dpi=150)
    plt.close()
    print(f"  Saved: {output_path / 'node_types.png'}")
    
    # 2. Edge type distribution
    fig, ax = plt.subplots(figsize=(12, 6))
    edge_names = []
    edge_counts = []
    for edge_type in data.edge_types:
        src, rel, dst = edge_type
        name = f"{src[:4]}→{rel}→{dst[:4]}"
        edge_names.append(name)
        edge_counts.append(data[edge_type].edge_index.size(1))
    
    ax.barh(edge_names, edge_counts, color='#3498db')
    ax.set_xlabel('Number of Edges')
    ax.set_title('Edge Types in Legal Graph')
    for i, (name, count) in enumerate(zip(edge_names, edge_counts)):
        ax.text(count + 10, i, str(count), va='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path / 'edge_types.png', dpi=150)
    plt.close()
    print(f"  Saved: {output_path / 'edge_types.png'}")
    
    # 3. Outcome distribution
    if hasattr(data["case"], "y"):
        fig, ax = plt.subplots(figsize=(8, 6))
        labels = data["case"].y
        bozma = (labels == 0).sum().item()
        onama = (labels == 1).sum().item()
        ax.pie([onama, bozma], labels=['ONAMA', 'BOZMA'], 
               autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'],
               explode=[0.02, 0.02])
        ax.set_title('Outcome Distribution')
        plt.savefig(output_path / 'outcome_dist.png', dpi=150)
        plt.close()
        print(f"  Saved: {output_path / 'outcome_dist.png'}")
    
    # 4. Similarity edge weight distribution (if exists)
    sim_edge = ("case", "similar", "case")
    if sim_edge in data.edge_types and hasattr(data[sim_edge], 'edge_weight'):
        fig, ax = plt.subplots(figsize=(10, 6))
        weights = data[sim_edge].edge_weight.numpy()
        ax.hist(weights, bins=50, color='#9b59b6', alpha=0.7, edgecolor='white')
        ax.set_xlabel('Cosine Similarity')
        ax.set_ylabel('Count')
        ax.set_title('Similarity Edge Weight Distribution')
        ax.axvline(weights.mean(), color='red', linestyle='--', label=f'Mean: {weights.mean():.3f}')
        ax.legend()
        plt.tight_layout()
        plt.savefig(output_path / 'similarity_weights.png', dpi=150)
        plt.close()
        print(f"  Saved: {output_path / 'similarity_weights.png'}")
    
    print(f"\n✅ All visualizations saved to: {output_path}")


def run_gnn_experiment(graph_file: str, embedding_name: str) -> dict:
    """Run GNN experiment and return results."""
    print_header("GNN TRAINING & EVALUATION")
    
    from src.graph import create_model, run_experiment, generate_report
    
    data = torch.load(graph_file, weights_only=False)
    
    results = run_experiment(
        data,
        gnn_types=["gat", "han", "hgt"],
        embedding_name=embedding_name,
        num_epochs=100,
        verbose=True,
    )
    
    # Generate report
    report_file = f"results/report_{embedding_name}_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
    report = generate_report(results, output_file=report_file)
    
    print("\n" + report)
    print(f"\n✅ Report saved to: {report_file}")
    
    return results


def run_full_pipeline(
    parsed_file: str = "data/parsed_gpt.json",
    embedding_model: str = "berturk-legal",
    similarity_k: int = 10,
    run_training: bool = True,
):
    """Run full analysis pipeline."""
    print("\n" + "█" * 70)
    print("  LEGAL GRAPH ANALYSIS - FULL PIPELINE")
    print("█" * 70)
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Parsed file: {parsed_file}")
    print(f"Embedding model: {embedding_model}")
    
    # 1. Analyze parsed data
    parsed_stats = analyze_parsed_data(parsed_file)
    
    # 2. Generate embeddings
    print_header("GENERATING EMBEDDINGS")
    
    with open(parsed_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    # Use ALL cases with statutes (for statute recommendation)
    # Outcome prediction will filter to ONAMA/BOZMA during training
    cases = [c for c in raw_data if c.get("statute_ids")]
    texts = [c.get("plaintiff_arguments", "") for c in cases]
    
    print(f"Generating embeddings for {len(texts)} cases with statutes...")
    
    from src.graph import generate_embeddings
    
    cache_file = f"data/embeddings_{embedding_model.replace('-', '_')}_new.pt"
    embeddings = generate_embeddings(
        texts,
        model=embedding_model,
        cache_file=cache_file,
    )
    
    # 3. Analyze embeddings
    emb_stats = analyze_embeddings(cache_file, parsed_file)
    
    # 4. Build graph
    print_header("BUILDING GRAPH")
    
    from src.graph import LegalGraphBuilder
    
    builder = LegalGraphBuilder(
        parsed_file,
        embeddings=embeddings,
        embedding_model=embedding_model,
    )
    
    data = builder.build(similarity_k=similarity_k, similarity_threshold=0.3)
    data = builder.create_train_test_split(data)
    
    graph_file = f"data/graph_{embedding_model.replace('-', '_')}_new.pt"
    torch.save(data, graph_file)
    print(f"\n✅ Graph saved to: {graph_file}")
    
    # 5. Analyze graph
    graph_stats = analyze_graph(graph_file)
    
    # 6. Visualizations
    visualize_graph(graph_file, "results/visualizations")
    
    # 7. Run GNN experiment
    if run_training:
        gnn_results = run_gnn_experiment(graph_file, embedding_model)
    else:
        gnn_results = None
    
    # Final summary
    print_header("PIPELINE COMPLETE")
    print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n📊 Summary:")
    print(f"  Cases: {graph_stats['n_cases']}")
    print(f"  Statutes: {graph_stats['n_statutes']}")
    print(f"  Chambers: {graph_stats['n_chambers']}")
    print(f"  Total edges: {sum(graph_stats['edge_stats'].values())}")
    
    if gnn_results:
        best_outcome = max(gnn_results["models"].items(), key=lambda x: x[1]["outcome"]["f1"])
        best_statute = max(gnn_results["models"].items(), key=lambda x: x[1]["statute"]["recall@10"])
        
        print(f"\n🏆 Best Model (Outcome): {best_outcome[0].upper()} - F1: {best_outcome[1]['outcome']['f1']:.4f}")
        print(f"🏆 Best Model (Statute): {best_statute[0].upper()} - R@10: {best_statute[1]['statute']['recall@10']:.4f}")
    
    print(f"\n📁 Output files:")
    print(f"  - {graph_file}")
    print(f"  - {cache_file}")
    print(f"  - results/visualizations/")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run full analysis pipeline")
    parser.add_argument("--parsed-file", default="data/parsed_gpt.json")
    parser.add_argument("--embedding", default="berturk-legal", 
                       choices=["berturk", "berturk-legal", "openai"])
    parser.add_argument("--similarity-k", type=int, default=10)
    parser.add_argument("--skip-training", action="store_true", 
                       help="Skip GNN training (just analyze)")
    
    args = parser.parse_args()
    
    run_full_pipeline(
        parsed_file=args.parsed_file,
        embedding_model=args.embedding,
        similarity_k=args.similarity_k,
        run_training=not args.skip_training,
    )

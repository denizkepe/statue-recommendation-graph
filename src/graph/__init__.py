"""
Legal Graph Analysis - Graph Module

This module provides tools for building and training GNNs on legal graphs.

Components:
- embeddings: Multi-model embedding generation (BERTurk, BERTurk-Legal, OpenAI)
- builder: Heterogeneous graph construction
- models: GNN architectures (GAT, HAN, HGT)
- trainer: Training and evaluation utilities
"""

from .embeddings import (
    generate_embeddings,
    compare_embeddings,
    get_embedding_dim,
    EMBEDDING_MODELS,
)

from .builder import (
    LegalGraphBuilder,
    build_legal_graph,
)

from .models import (
    HeteroGAT,
    HAN,
    HGT,
    LegalGNN,
    create_model,
)

from .trainer import (
    train_outcome_model,
    train_statute_model,
    run_experiment,
    generate_report,
)

__all__ = [
    # Embeddings
    "generate_embeddings",
    "compare_embeddings",
    "get_embedding_dim",
    "EMBEDDING_MODELS",
    # Graph building
    "LegalGraphBuilder",
    "build_legal_graph",
    # Models
    "HeteroGAT",
    "HAN",
    "HGT",
    "LegalGNN",
    "create_model",
    # Training
    "train_outcome_model",
    "train_statute_model",
    "run_experiment",
    "generate_report",
]

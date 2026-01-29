"""
Heterogeneous Legal Graph Builder.

Builds a PyTorch Geometric HeteroData object with:
- Node types: case, statute, chamber
- Edge types: cites, belongs_to, co_cited, similar

Key features:
1. Text embeddings as case node features
2. Weighted similarity edges based on cosine similarity
3. TF-IDF weighted citation edges
4. Co-citation patterns between statutes
"""

import json
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple, Optional, Literal

try:
    from torch_geometric.data import HeteroData
    import torch_geometric.transforms as T
except ImportError:
    raise ImportError("Please install torch-geometric: pip install torch-geometric")


class LegalGraphBuilder:
    """
    Builds a heterogeneous graph for legal judgment prediction.
    
    Graph Structure:
    ================
    
    NODE TYPES:
    -----------
    1. case: Court decisions (plaintiff's arguments as features)
       - Features: BERT embeddings (768 or 1536 dim)
       - Labels: Outcome (ONAMA=1, BOZMA=0)
       
    2. statute: Law articles (e.g., 4857-25 = İş Kanunu madde 25)
       - Features: Citation degree + learnable embeddings
       
    3. chamber: Court chambers (e.g., 9. Hukuk Dairesi)
       - Features: One-hot encoding
    
    EDGE TYPES:
    -----------
    1. (case, cites, statute): Citation relationship
       - Weight: TF-IDF score (rare statutes get higher weight)
       
    2. (case, belongs_to, chamber): Court specialization
       
    3. (statute, co_cited, statute): Statutes cited together
       - Weight: Co-citation frequency
       
    4. (case, similar, case): Similar plaintiff arguments
       - Weight: Cosine similarity of embeddings
       - Created from text embeddings
       
    META-PATHS:
    -----------
    - Case → Statute → Case: Cases citing same statutes
    - Case → Chamber → Case: Cases in same court division
    - Case → Similar → Case: Cases with similar text
    """
    
    def __init__(
        self,
        parsed_file: str,
        embeddings: torch.Tensor = None,
        embedding_model: str = "berturk-legal",
    ):
        """
        Initialize graph builder.
        
        Args:
            parsed_file: Path to parsed JSON file
            embeddings: Pre-computed embeddings (optional)
            embedding_model: Model used for embeddings (for metadata)
        """
        self.parsed_file = Path(parsed_file)
        self.embeddings = embeddings
        self.embedding_model = embedding_model
        
        # Load and filter data
        with open(self.parsed_file, 'r', encoding='utf-8') as f:
            self.raw_data = json.load(f)
        
        # Filter to valid cases
        self.cases = [
            c for c in self.raw_data
            if c.get("outcome") in ["ONAMA", "BOZMA"] and c.get("statute_ids")
        ]
        
        print(f"Loaded {len(self.cases)} valid cases")
        self._build_mappings()
    
    def _build_mappings(self):
        """Build ID to index mappings for all node types."""
        # Case mapping
        self.case_ids = [c["id"] for c in self.cases]
        self.case_to_idx = {cid: i for i, cid in enumerate(self.case_ids)}
        
        # Statute mapping
        statute_set = set()
        for c in self.cases:
            statute_set.update(c.get("statute_ids", []))
        self.statute_ids = sorted(statute_set)
        self.statute_to_idx = {s: i for i, s in enumerate(self.statute_ids)}
        
        # Chamber mapping
        chamber_set = {c.get("chamber") for c in self.cases if c.get("chamber")}
        self.chamber_ids = sorted(chamber_set)
        self.chamber_to_idx = {ch: i for i, ch in enumerate(self.chamber_ids)}
        
        # Case type mapping (semi-dynamic from extracted data)
        case_type_set = {c.get("case_type_enum") for c in self.cases if c.get("case_type_enum")}
        # Also include types from raw_data for cases not in filtered set
        for c in self.raw_data:
            if c.get("case_type_enum"):
                case_type_set.add(c.get("case_type_enum"))
        self.case_type_ids = sorted(case_type_set) if case_type_set else []
        self.case_type_to_idx = {ct: i for i, ct in enumerate(self.case_type_ids)}
        
        print(f"  Cases: {len(self.case_ids)}")
        print(f"  Statutes: {len(self.statute_ids)}")
        print(f"  Chambers: {len(self.chamber_ids)}")
        print(f"  Case Types: {len(self.case_type_ids)}")
    
    def build_citation_edges(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build case → statute citation edges with TF-IDF weights.
        
        TF-IDF weighting: Rare statutes get higher weight because
        they are more specific/informative.
        """
        src, dst, weights = [], [], []
        
        # Count statute frequencies for IDF
        statute_freq = defaultdict(int)
        for c in self.cases:
            for s in c.get("statute_ids", []):
                if s in self.statute_to_idx:
                    statute_freq[s] += 1
        
        n_cases = len(self.cases)
        
        for c in self.cases:
            c_idx = self.case_to_idx[c["id"]]
            for s in c.get("statute_ids", []):
                if s in self.statute_to_idx:
                    src.append(c_idx)
                    dst.append(self.statute_to_idx[s])
                    # IDF weight
                    idf = np.log(n_cases / (1 + statute_freq[s]))
                    weights.append(max(0.1, idf))  # Minimum weight 0.1
        
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_weight = torch.tensor(weights, dtype=torch.float32)
        
        print(f"  Citation edges: {len(src)}")
        return edge_index, edge_weight
    
    def build_similarity_edges(
        self,
        top_k: int = 10,
        threshold: float = 0.3,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build case ↔ case similarity edges based on embedding similarity.
        
        This is the KEY innovation: cases with similar plaintiff arguments
        should cite similar statutes. By creating explicit similarity edges,
        the GNN can propagate this information during message passing.
        
        Args:
            top_k: Number of most similar cases to connect
            threshold: Minimum cosine similarity to create edge
            
        Returns:
            (edge_index, edge_weight)
        """
        if self.embeddings is None:
            print("  Similarity edges: SKIPPED (no embeddings)")
            return None, None
        
        # Compute pairwise cosine similarities
        emb_np = self.embeddings.numpy()
        sims = cosine_similarity(emb_np)
        
        src, dst, weights = [], [], []
        
        for i in range(len(self.case_ids)):
            # Get similarity scores (exclude self)
            sim_scores = sims[i].copy()
            sim_scores[i] = -1
            
            # Get top-k most similar
            top_indices = np.argsort(sim_scores)[-top_k:]
            
            for j in top_indices:
                if sim_scores[j] >= threshold:
                    src.append(i)
                    dst.append(j)
                    weights.append(sim_scores[j])
        
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_weight = torch.tensor(weights, dtype=torch.float32)
        
        print(f"  Similarity edges: {len(src)} (top-{top_k}, threshold={threshold})")
        print(f"    Mean similarity: {edge_weight.mean():.3f}")
        print(f"    Min/Max: {edge_weight.min():.3f} / {edge_weight.max():.3f}")
        
        return edge_index, edge_weight
    
    def build_cocitation_edges(self, min_count: int = 2) -> torch.Tensor:
        """
        Build statute ↔ statute co-citation edges.
        
        Two statutes are connected if they are cited together
        in at least min_count cases.
        """
        co_cite = defaultdict(int)
        
        for c in self.cases:
            statutes = [s for s in c.get("statute_ids", []) if s in self.statute_to_idx]
            for i, s1 in enumerate(statutes):
                for s2 in statutes[i + 1:]:
                    pair = tuple(sorted([s1, s2]))
                    co_cite[pair] += 1
        
        src, dst = [], []
        for (s1, s2), cnt in co_cite.items():
            if cnt >= min_count:
                i1, i2 = self.statute_to_idx[s1], self.statute_to_idx[s2]
                # Bidirectional
                src.extend([i1, i2])
                dst.extend([i2, i1])
        
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        print(f"  Co-citation edges: {len(src)}")
        
        return edge_index
    
    def build(
        self,
        similarity_k: int = 10,
        similarity_threshold: float = 0.3,
        add_reverse: bool = True,
    ) -> HeteroData:
        """
        Build complete HeteroData object.
        
        Args:
            similarity_k: Top-k similar cases to connect
            similarity_threshold: Minimum similarity for edge
            add_reverse: Add reverse edges (for undirected message passing)
            
        Returns:
            PyTorch Geometric HeteroData object
        """
        print("\n" + "=" * 50)
        print("BUILDING HETEROGENEOUS GRAPH")
        print("=" * 50)
        
        data = HeteroData()
        
        # === NODE FEATURES ===
        print("\nNode features:")
        
        # Case: Use embeddings or random
        if self.embeddings is not None:
            data["case"].x = self.embeddings
            print(f"  case: {data['case'].x.shape} ({self.embedding_model} embeddings)")
        else:
            data["case"].x = torch.randn(len(self.case_ids), 768)
            print(f"  case: {data['case'].x.shape} (random, no embeddings provided)")
        
        # Statute: Degree + learnable
        statute_degree = torch.zeros(len(self.statute_ids))
        for c in self.cases:
            for s in c.get("statute_ids", []):
                if s in self.statute_to_idx:
                    statute_degree[self.statute_to_idx[s]] += 1
        
        statute_degree = torch.log1p(statute_degree).unsqueeze(1)
        data["statute"].x = torch.cat([
            statute_degree,
            torch.randn(len(self.statute_ids), 63)
        ], dim=1)
        print(f"  statute: {data['statute'].x.shape}")
        
        # Chamber: One-hot
        data["chamber"].x = torch.eye(len(self.chamber_ids))
        print(f"  chamber: {data['chamber'].x.shape}")
        
        # Case Type: One-hot (for meta-path learning)
        if len(self.case_type_ids) > 0:
            data["case_type"].x = torch.eye(len(self.case_type_ids))
            print(f"  case_type: {data['case_type'].x.shape}")
        
        # Node IDs
        data["case"].node_ids = self.case_ids
        data["statute"].node_ids = self.statute_ids
        data["chamber"].node_ids = self.chamber_ids
        if len(self.case_type_ids) > 0:
            data["case_type"].node_ids = self.case_type_ids
        
        # === LABELS ===
        labels = [0 if c.get("outcome") == "BOZMA" else 1 for c in self.cases]
        data["case"].y = torch.tensor(labels, dtype=torch.long)
        data["case"].labeled_mask = torch.ones(len(labels), dtype=torch.bool)
        
        n_bozma = sum(1 for l in labels if l == 0)
        n_onama = sum(1 for l in labels if l == 1)
        print(f"\nLabels: BOZMA={n_bozma}, ONAMA={n_onama}")
        
        # === EDGES ===
        print("\nEdges:")
        
        # 1. Citation edges
        cite_edge_index, cite_weight = self.build_citation_edges()
        data["case", "cites", "statute"].edge_index = cite_edge_index
        data["case", "cites", "statute"].edge_weight = cite_weight
        
        # 2. Chamber edges
        src, dst = [], []
        for c in self.cases:
            ch = c.get("chamber")
            if ch and ch in self.chamber_to_idx:
                src.append(self.case_to_idx[c["id"]])
                dst.append(self.chamber_to_idx[ch])
        data["case", "belongs_to", "chamber"].edge_index = torch.tensor([src, dst], dtype=torch.long)
        print(f"  Chamber edges: {len(src)}")
        
        # 3. Case type edges (case -> case_type)
        if len(self.case_type_ids) > 0:
            src, dst = [], []
            for c in self.cases:
                ct = c.get("case_type_enum")
                if ct and ct in self.case_type_to_idx:
                    src.append(self.case_to_idx[c["id"]])
                    dst.append(self.case_type_to_idx[ct])
            if src:
                data["case", "has_type", "case_type"].edge_index = torch.tensor([src, dst], dtype=torch.long)
                print(f"  Case type edges: {len(src)}")
        
        # 4. Co-citation edges
        data["statute", "co_cited", "statute"].edge_index = self.build_cocitation_edges()
        
        # 5. Similarity edges
        sim_edge_index, sim_weight = self.build_similarity_edges(
            top_k=similarity_k,
            threshold=similarity_threshold
        )
        if sim_edge_index is not None:
            data["case", "similar", "case"].edge_index = sim_edge_index
            data["case", "similar", "case"].edge_weight = sim_weight
        
        # Add reverse edges for undirected message passing
        if add_reverse:
            data = T.ToUndirected()(data)
        
        print("\n" + "=" * 50)
        print("GRAPH SUMMARY")
        print("=" * 50)
        print(data)
        
        return data
    
    def create_train_test_split(
        self,
        data: HeteroData,
        test_ratio: float = 0.2,
        val_ratio: float = 0.1,
        seed: int = 42,
    ) -> HeteroData:
        """
        Create train/val/test splits for both node and link prediction.
        
        Args:
            data: HeteroData object
            test_ratio: Fraction for testing
            val_ratio: Fraction for validation
            seed: Random seed
            
        Returns:
            Data with train_mask, val_mask, test_mask attributes
        """
        torch.manual_seed(seed)
        
        # Node split (for outcome prediction)
        n_cases = len(self.case_ids)
        perm = torch.randperm(n_cases)
        
        n_test = int(n_cases * test_ratio)
        n_val = int(n_cases * val_ratio)
        
        data["case"].train_mask = torch.zeros(n_cases, dtype=torch.bool)
        data["case"].val_mask = torch.zeros(n_cases, dtype=torch.bool)
        data["case"].test_mask = torch.zeros(n_cases, dtype=torch.bool)
        
        data["case"].train_mask[perm[n_test + n_val:]] = True
        data["case"].val_mask[perm[n_test:n_test + n_val]] = True
        data["case"].test_mask[perm[:n_test]] = True
        
        print(f"\nNode split: train={data['case'].train_mask.sum().item()}, "
              f"val={data['case'].val_mask.sum().item()}, "
              f"test={data['case'].test_mask.sum().item()}")
        
        # Edge split (for link prediction)
        edge_type = ("case", "cites", "statute")
        if edge_type in data.edge_types:
            edge_index = data[edge_type].edge_index
            n_edges = edge_index.size(1)
            
            perm = torch.randperm(n_edges)
            n_test_e = int(n_edges * test_ratio)
            n_val_e = int(n_edges * val_ratio)
            
            data[edge_type].train_edge_index = edge_index[:, perm[n_test_e + n_val_e:]]
            data[edge_type].val_edge_index = edge_index[:, perm[n_test_e:n_test_e + n_val_e]]
            data[edge_type].test_edge_index = edge_index[:, perm[:n_test_e]]
            
            print(f"Edge split: train={data[edge_type].train_edge_index.size(1)}, "
                  f"val={data[edge_type].val_edge_index.size(1)}, "
                  f"test={data[edge_type].test_edge_index.size(1)}")
        
        return data


def build_legal_graph(
    parsed_file: str,
    embedding_model: Literal["berturk", "berturk-legal", "openai"] = "berturk-legal",
    embedding_cache: str = None,
    similarity_k: int = 10,
    similarity_threshold: float = 0.3,
    output_file: str = None,
) -> HeteroData:
    """
    Convenience function to build complete legal graph.
    
    Args:
        parsed_file: Path to parsed JSON
        embedding_model: Model for case embeddings
        embedding_cache: Path to cache embeddings
        similarity_k: Top-k similar cases
        similarity_threshold: Min similarity for edge
        output_file: Path to save graph
        
    Returns:
        HeteroData object
    """
    from .embeddings import generate_embeddings
    
    # Load cases
    import json
    with open(parsed_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    # Filter valid cases
    cases = [
        c for c in raw_data
        if c.get("outcome") in ["ONAMA", "BOZMA"] and c.get("statute_ids")
    ]
    
    # Get texts for embedding
    texts = [c.get("plaintiff_arguments", "") or c.get("claim_summary", "") for c in cases]
    
    # Generate embeddings
    if embedding_cache is None:
        embedding_cache = f"data/embeddings_{embedding_model.replace('-', '_')}.pt"
    
    embeddings = generate_embeddings(
        texts,
        model=embedding_model,
        cache_file=embedding_cache,
    )
    
    # Build graph
    builder = LegalGraphBuilder(
        parsed_file,
        embeddings=embeddings,
        embedding_model=embedding_model,
    )
    
    data = builder.build(
        similarity_k=similarity_k,
        similarity_threshold=similarity_threshold,
    )
    
    data = builder.create_train_test_split(data)
    
    # Save
    if output_file:
        torch.save(data, output_file)
        print(f"\nGraph saved to {output_file}")
    
    return data


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build legal heterogeneous graph")
    parser.add_argument("--input", required=True, help="Parsed JSON file")
    parser.add_argument("--output", default="data/legal_graph.pt", help="Output graph file")
    parser.add_argument("--embedding-model", default="berturk-legal",
                       choices=["berturk", "berturk-legal", "openai"])
    parser.add_argument("--similarity-k", type=int, default=10)
    parser.add_argument("--similarity-threshold", type=float, default=0.3)
    args = parser.parse_args()
    
    build_legal_graph(
        parsed_file=args.input,
        embedding_model=args.embedding_model,
        similarity_k=args.similarity_k,
        similarity_threshold=args.similarity_threshold,
        output_file=args.output,
    )

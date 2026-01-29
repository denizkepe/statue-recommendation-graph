"""
GNN Models for Legal Judgment Prediction.

Three heterogeneous graph neural networks:
1. HeteroGAT - Graph Attention Networks with type-specific attention
2. HAN - Hierarchical Attention Networks with meta-path aggregation  
3. HGT - Heterogeneous Graph Transformer with type-specific transformations

All models support:
- Node classification (outcome prediction: ONAMA vs BOZMA)
- Link prediction (statute recommendation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, HGTConv, Linear, HANConv, HeteroConv
from typing import Dict, List, Tuple, Literal


class HeteroGAT(nn.Module):
    """
    Heterogeneous Graph Attention Network.
    
    Architecture:
    =============
    For each edge type, a separate GATv2Conv is used.
    HeteroConv aggregates messages from all edge types.
    
    Key Features:
    - Multi-head attention learns neighbor importance
    - Different attention patterns per edge type
    - Similarity edges enable learning from similar cases
    
    Message Passing:
    ----------------
    For case node i:
      h_i = Σ_j∈N(i) α_ij · W·h_j
      
    where α_ij is learned attention based on node features.
    
    Similar cases (high α) contribute more to the final embedding.
    """
    
    def __init__(
        self,
        in_channels: Dict[str, int],
        hidden_channels: int = 128,
        out_channels: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.3,
        metadata: Tuple = None,
    ):
        super().__init__()
        
        self.num_layers = num_layers
        
        # Project each node type to common dimension
        self.lin_dict = nn.ModuleDict()
        for node_type, in_dim in in_channels.items():
            self.lin_dict[node_type] = Linear(in_dim, hidden_channels)
        
        # Create HeteroConv layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            conv_dict = {}
            for edge_type in metadata[1]:
                conv_dict[edge_type] = GATv2Conv(
                    hidden_channels,
                    hidden_channels // num_heads,
                    heads=num_heads,
                    concat=True,
                    dropout=dropout,
                    add_self_loops=False,
                )
            self.convs.append(HeteroConv(conv_dict, aggr='sum'))
        
        self.out_lin = Linear(hidden_channels, out_channels)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x_dict, edge_index_dict) -> Dict[str, torch.Tensor]:
        # Project inputs
        h_dict = {}
        for node_type, x in x_dict.items():
            if node_type in self.lin_dict:
                h_dict[node_type] = self.lin_dict[node_type](x)
            else:
                h_dict[node_type] = x
        
        # Apply GNN layers
        for conv in self.convs:
            h_dict = conv(h_dict, edge_index_dict)
            h_dict = {k: F.elu(self.dropout(v)) for k, v in h_dict.items()}
        
        return {k: self.out_lin(v) for k, v in h_dict.items()}


class HAN(nn.Module):
    """
    Hierarchical Attention Network.
    
    Architecture:
    =============
    Two-level attention mechanism:
    1. Node-level: Aggregate neighbors within each meta-path
    2. Meta-path-level: Learn importance of different meta-paths
    
    Meta-Paths for Legal Graph:
    ---------------------------
    - Case → Statute → Case: Cases citing same statutes
    - Case → Chamber → Case: Cases in same court division
    - Case → Similar → Case: Cases with similar text
    
    Why This Works:
    ---------------
    The model learns that for labor cases, the Case→Chamber→Case
    meta-path (9. Hukuk Dairesi) is more informative than
    for general civil cases.
    """
    
    def __init__(
        self,
        in_channels: Dict[str, int],
        hidden_channels: int = 128,
        out_channels: int = 64,
        num_heads: int = 4,
        dropout: float = 0.3,
        metadata: Tuple = None,
    ):
        super().__init__()
        
        # Project each node type
        self.lin_dict = nn.ModuleDict()
        for node_type, in_dim in in_channels.items():
            self.lin_dict[node_type] = Linear(in_dim, hidden_channels)
        
        # HAN layers
        self.conv1 = HANConv(hidden_channels, hidden_channels, metadata, heads=num_heads, dropout=dropout)
        self.conv2 = HANConv(hidden_channels, hidden_channels, metadata, heads=num_heads, dropout=dropout)
        
        self.out_lin = Linear(hidden_channels, out_channels)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x_dict, edge_index_dict) -> Dict[str, torch.Tensor]:
        h_dict = {k: self.lin_dict[k](v) for k, v in x_dict.items() if k in self.lin_dict}
        
        h_dict = self.conv1(h_dict, edge_index_dict)
        h_dict = {k: F.elu(self.dropout(v)) for k, v in h_dict.items()}
        
        h_dict = self.conv2(h_dict, edge_index_dict)
        h_dict = {k: F.elu(v) for k, v in h_dict.items()}
        
        return {k: self.out_lin(v) for k, v in h_dict.items()}


class HGT(nn.Module):
    """
    Heterogeneous Graph Transformer.
    
    Architecture:
    =============
    Type-specific transformations:
    - Different W_Q, W_K, W_V for each node type
    - Different attention for each edge type
    
    Key Innovation:
    ---------------
    For edge (case_i, similar, case_j):
      Q = W_Q^case · h_i
      K = W_K^case · h_j  
      V = W_V^similar · h_j
      
    The model learns that similarity edges should have
    different importance than citation edges.
    
    Best For:
    ---------
    Complex heterogeneous graphs with many node/edge types.
    """
    
    def __init__(
        self,
        in_channels: Dict[str, int],
        hidden_channels: int = 128,
        out_channels: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.3,
        metadata: Tuple = None,
    ):
        super().__init__()
        
        self.num_layers = num_layers
        
        self.lin_dict = nn.ModuleDict()
        for node_type, in_dim in in_channels.items():
            self.lin_dict[node_type] = Linear(in_dim, hidden_channels)
        
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(HGTConv(hidden_channels, hidden_channels, metadata, heads=num_heads))
        
        self.out_lin = Linear(hidden_channels, out_channels)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x_dict, edge_index_dict) -> Dict[str, torch.Tensor]:
        h_dict = {k: self.lin_dict[k](v) for k, v in x_dict.items() if k in self.lin_dict}
        
        for conv in self.convs:
            h_dict = conv(h_dict, edge_index_dict)
            h_dict = {k: F.relu(self.dropout(v)) for k, v in h_dict.items()}
        
        return {k: self.out_lin(v) for k, v in h_dict.items()}


class NodeClassifier(nn.Module):
    """Predict outcome (ONAMA/BOZMA) from case embeddings."""
    
    def __init__(self, in_channels: int, num_classes: int = 2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )
    
    def forward(self, x):
        return self.mlp(x)


class LinkPredictor(nn.Module):
    """Predict citation probability between case and statute."""
    
    def __init__(self, in_channels: int):
        super().__init__()
        self.lin_case = nn.Linear(in_channels, 64)
        self.lin_statute = nn.Linear(in_channels, 64)
        self.decoder = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )
    
    def forward(self, case_emb, statute_emb, edge_index):
        src = self.lin_case(case_emb[edge_index[0]])
        dst = self.lin_statute(statute_emb[edge_index[1]])
        return self.decoder(torch.cat([src, dst], dim=-1)).squeeze(-1)
    
    def predict_all(self, case_emb, statute_emb):
        """Predict scores for all (case, statute) pairs."""
        case_proj = self.lin_case(case_emb)
        statute_proj = self.lin_statute(statute_emb)
        
        scores = torch.zeros(case_proj.size(0), statute_proj.size(0))
        for i in range(case_proj.size(0)):
            feat = torch.cat([
                case_proj[i].unsqueeze(0).expand(statute_proj.size(0), -1),
                statute_proj
            ], dim=-1)
            scores[i] = self.decoder(feat).squeeze(-1)
        return scores


class LegalGNN(nn.Module):
    """
    Complete model: GNN encoder + prediction heads.
    
    Usage:
        model = LegalGNN("gat", in_channels, metadata=metadata)
        embeddings = model(data)
        outcome_logits = model.predict_outcome(embeddings)
        statute_scores = model.recommend_statutes(embeddings)
    """
    
    def __init__(
        self,
        gnn_type: Literal["gat", "han", "hgt"],
        in_channels: Dict[str, int],
        hidden_channels: int = 128,
        out_channels: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.3,
        metadata: Tuple = None,
    ):
        super().__init__()
        
        self.gnn_type = gnn_type
        
        args = dict(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_heads=num_heads,
            dropout=dropout,
            metadata=metadata,
        )
        
        if gnn_type == "gat":
            args["num_layers"] = num_layers
            self.encoder = HeteroGAT(**args)
        elif gnn_type == "han":
            self.encoder = HAN(**args)
        elif gnn_type == "hgt":
            args["num_layers"] = num_layers
            self.encoder = HGT(**args)
        else:
            raise ValueError(f"Unknown GNN type: {gnn_type}")
        
        self.node_clf = NodeClassifier(out_channels)
        self.link_pred = LinkPredictor(out_channels)
    
    def forward(self, data) -> Dict[str, torch.Tensor]:
        """Get node embeddings."""
        # Include all node types that exist in the graph
        node_types = ["case", "statute", "chamber", "case_type"]
        x_dict = {k: data[k].x for k in node_types if k in data.node_types}
        return self.encoder(x_dict, data.edge_index_dict)
    
    def predict_outcome(self, embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Predict outcome for cases."""
        return self.node_clf(embeddings["case"])
    
    def predict_links(self, embeddings: Dict[str, torch.Tensor], edge_index) -> torch.Tensor:
        """Predict citation probability for given edges."""
        return self.link_pred(embeddings["case"], embeddings["statute"], edge_index)
    
    def recommend_statutes(self, embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Get statute recommendation scores for all cases."""
        return self.link_pred.predict_all(embeddings["case"], embeddings["statute"])


def create_model(
    gnn_type: Literal["gat", "han", "hgt"],
    data,
    hidden_channels: int = 128,
    out_channels: int = 64,
    num_heads: int = 4,
    num_layers: int = 2,
) -> LegalGNN:
    """
    Convenience function to create model from data.
    
    Args:
        gnn_type: One of "gat", "han", "hgt"
        data: HeteroData object
        hidden_channels: Hidden dimension
        out_channels: Output dimension
        
    Returns:
        LegalGNN model
    """
    in_channels = {
        "case": data["case"].x.size(1),
        "statute": data["statute"].x.size(1),
        "chamber": data["chamber"].x.size(1),
    }
    
    # Add case_type if present in graph
    if "case_type" in data.node_types:
        in_channels["case_type"] = data["case_type"].x.size(1)
    
    return LegalGNN(
        gnn_type=gnn_type,
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        num_heads=num_heads,
        num_layers=num_layers,
        metadata=data.metadata(),
    )

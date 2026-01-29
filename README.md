# 📜 Legal Graph Analysis: Statute Recommendation & Outcome Prediction using GNNs

A comprehensive Graph Neural Network (GNN) system for analyzing Turkish Yargıtay (Supreme Court) decisions. The system predicts which statutes should be cited and predicts case outcomes based on plaintiff arguments.

## 🎯 Project Objectives

This project addresses two key legal AI tasks:

1. **Statute Recommendation**: Given a lawyer's petition (plaintiff arguments), recommend the most relevant statutes (kanun maddeleri) to cite.

2. **Outcome Prediction**: Predict whether the court will rule ONAMA (affirm) or BOZMA (reverse) the lower court's decision.

### Real-World Use Case

```
┌─────────────────────────────────────────────────────────────────┐
│                    LAWYER'S WORKFLOW                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  INPUT:                                                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Petition Text (Dava Dilekçesi):                        │   │
│  │  "Müvekkilim 5 yıldır davalı şirkette çalışmakta olup,  │   │
│  │   iş akdi haksız yere feshedilmiştir. Kıdem ve ihbar    │   │
│  │   tazminatı talep etmekteyiz..."                        │   │
│  │                                                          │   │
│  │  Target Chamber: 9. Hukuk Dairesi (Labor Law)           │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                              ▼                                  │
│                        GNN SYSTEM                               │
│                              │                                  │
│                              ▼                                  │
│  OUTPUT:                                                        │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Recommended Statutes:                                   │   │
│  │  • 4857-25 (İş Kanunu, haklı fesih)                     │   │
│  │  • 4857-21/3 (İş Kanunu, kıdem tazminatı)               │   │
│  │  • 4857-17 (İş Kanunu, ihbar tazminatı)                 │   │
│  │                                                          │   │
│  │  Predicted Outcome: BOZMA (72% confidence)              │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📊 Latest Results (December 2024)

### Dataset Statistics

| Metric | Value |
|--------|-------|
| **Total Parsed Cases** | 2,481 |
| **Valid for Outcome Prediction** | 699 (ONAMA/BOZMA cases) |
| **Valid for Statute Recommendation** | 2,260 (cases with statutes) |
| **Unique Statutes** | 298 |
| **Chambers** | 28 |
| **Case Types** | 8 |
| **Total Statute Citations** | 4,880 |
| **Average Statutes per Case** | 1.97 |

### Outcome Distribution

```
GOREVSIZLIK (Jurisdiction)  ██████████████████████████████████  1,689 (68.1%)
ONAMA (Affirm)              █████████                            462 (18.6%)
BOZMA (Reverse)             █████                                255 (10.3%)
GERI_CEVIRME (Return)       █                                     69 (2.8%)
UNKNOWN                                                             6 (0.2%)
```

### Case Type Distribution

| Case Type | Count | Percentage |
|-----------|-------|------------|
| ISE_IADE (Re-employment) | 1,169 | 47.1% |
| ALACAK_TAZMINAT (Compensation) | 549 | 22.1% |
| GOREV_UYUSMAZLIGI (Jurisdiction Dispute) | 383 | 15.4% |
| IS_KAZASI (Work Accident) | 189 | 7.6% |
| HIZMET_TESPITI (Service Determination) | 179 | 7.2% |
| Other | 12 | 0.6% |

### Most Cited Statutes

| Statute | Count | Description |
|---------|-------|-------------|
| 5521-1 | 979 | İş Mahkemeleri Kanunu (Labor Courts Law) |
| 4857-18 | 461 | İş Kanunu (Labor Law) - Employment Security |
| 4857-19 | 456 | İş Kanunu - Reinstatement Procedure |
| 4857-20 | 456 | İş Kanunu - Reinstatement Litigation |
| 6100-22 | 359 | HMK (Civil Procedure) - Jurisdiction |
| 4857-25 | 281 | İş Kanunu - Justified Termination |
| 6100-21 | 244 | HMK - Venue |
| 4857-21/3 | 164 | İş Kanunu - Severance Pay |

### GNN Model Performance

#### Outcome Prediction (ONAMA vs BOZMA)

| Model | Accuracy | F1 Score | Precision | Recall |
|-------|----------|----------|-----------|--------|
| **HGT** | **0.7194** | **0.7133** | **0.7110** | **0.7194** |
| GAT | 0.6619 | 0.5272 | 0.4381 | 0.6619 |
| HAN | 0.6619 | 0.5272 | 0.4381 | 0.6619 |

#### Statute Recommendation

| Model | P@5 | R@5 | P@10 | R@10 | MRR |
|-------|-----|-----|------|------|-----|
| **HGT** | **0.1099** | **0.4621** | **0.0813** | **0.6819** | **0.3247** |
| GAT | 0.1099 | 0.4621 | 0.0703 | 0.5885 | 0.3180 |
| HAN | 0.0659 | 0.2875 | 0.0678 | 0.5647 | 0.2305 |

**Key Findings**:
- **HGT (Heterogeneous Graph Transformer)** achieves the best performance on both tasks
- **71.33% F1 score** for outcome prediction with 699 ONAMA/BOZMA cases
- **68.19% Recall@10** for statute recommendation, meaning ~68% of correct statutes appear in top-10
- Legal-domain BERT embeddings (BERTurk-Legal) provide effective case representations

---

## 🏗️ System Architecture

### Complete Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         DATA COLLECTION                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Yargıtay API (karararama.yargitay.gov.tr)                             │
│                     │                                                   │
│                     ▼                                                   │
│              Scraper Module (src/scrape/)                              │
│                     │                                                   │
│                     ▼                                                   │
│           data/raw/*.json (2,707 decision files)                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         PARSING PIPELINE                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Raw Text ──▶ LLM Parser (GPT-4o-mini) with Retry Logic                │
│                     │                                                   │
│                     ▼                                                   │
│  EXTRACTION PROMPT (Turkish):                                           │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Alanlar:                                                        │   │
│  │  • plaintiff_arguments: Davacı/vekili iddia ve talepleri        │   │
│  │  • case_type_enum: ISE_IADE | ALACAK_TAZMINAT | IS_KAZASI |     │   │
│  │                    HIZMET_TESPITI | GOREV_UYUSMAZLIGI           │   │
│  │  • outcome: ONAMA | BOZMA | GOREVSIZLIK | GERI_CEVIRME          │   │
│  │  • chamber: "9. Hukuk Dairesi", "21. Hukuk Dairesi", ...       │   │
│  │  • statute_ids: ["4857-25", "6100-22", "5521-1", ...]          │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                     │                                                   │
│  Features:                                                              │
│  • Exponential backoff retry for rate limits (429 errors)              │
│  • Error document filtering (skip malformed files)                     │
│  • Parallel processing with configurable workers                       │
│                     │                                                   │
│                     ▼                                                   │
│           data/parsed_all.json (2,481 valid cases)                     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       EMBEDDING GENERATION                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Plaintiff Arguments ──▶ BERT Encoder ──▶ 768-dim Embeddings           │
│                                                                         │
│  Supported Models:                                                      │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ berturk-legal (RECOMMENDED)                                      │   │
│  │   └── KocLab-Bilkent/BERTurk-Legal                              │   │
│  │       Pre-trained on Turkish legal texts                         │   │
│  │       768-dimensional embeddings                                 │   │
│  │                                                                   │   │
│  │ openai                                                            │   │
│  │   └── text-embedding-3-small                                    │   │
│  │       1536-dimensional embeddings                                │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                     │                                                   │
│  Caching: Embeddings are cached to data/embeddings_*.pt               │
│                     │                                                   │
│                     ▼                                                   │
│           data/embeddings_berturk_legal_new.pt                          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   HETEROGENEOUS GRAPH CONSTRUCTION                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  NODE TYPES (4 types):                                                  │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ CASE (2,260 total nodes, 699 labeled)                             │  │
│  │   Features: BERT embeddings (768-dim)                            │  │
│  │   Labels: ONAMA=444, BOZMA=255 (for binary classification)      │  │
│  │                                                                   │  │
│  │ STATUTE (252 nodes)                                               │  │
│  │   Features: Learnable embeddings (64-dim)                        │  │
│  │   Examples: 5521-1, 4857-18, 4857-19, 6100-22                   │  │
│  │                                                                   │  │
│  │ CHAMBER (25 nodes)                                                │  │
│  │   Features: One-hot encoding (25-dim)                            │  │
│  │   Examples: 21. Hukuk Dairesi, 9. Hukuk Dairesi, etc.           │  │
│  │                                                                   │  │
│  │ CASE_TYPE (8 nodes)                                               │  │
│  │   Features: One-hot encoding (8-dim)                             │  │
│  │   Values: ISE_IADE, ALACAK_TAZMINAT, GOREV_UYUSMAZLIGI,         │  │
│  │           IS_KAZASI, HIZMET_TESPITI, etc.                       │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  EDGE TYPES (8 types including reverse edges):                          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ case ──[cites]──▶ statute (1,655 edges)                          │  │
│  │   Weight: TF-IDF score (rare statutes = higher weight)          │  │
│  │   Mean weight: 2.8040, Range: [1.07, 5.86]                      │  │
│  │                                                                   │  │
│  │ case ──[belongs_to]──▶ chamber (699 edges)                       │  │
│  │   Which court division heard this case                          │  │
│  │                                                                   │  │
│  │ case ──[has_type]──▶ case_type (699 edges)                       │  │
│  │   Case category classification from LLM parsing                 │  │
│  │   Enables Case → CaseType → Case message passing                │  │
│  │                                                                   │  │
│  │ statute ◀──[co_cited]──▶ statute (248 edges)                    │  │
│  │   Statutes frequently cited together in same cases              │  │
│  │                                                                   │  │
│  │ case ◀──[similar]──▶ case (13,036 edges)                        │  │
│  │   Weight: Cosine similarity of embeddings                       │  │
│  │   Connected if similarity > 0.3 (top-10 per case)               │  │
│  │   Mean similarity: 0.978, Min: 0.850                            │  │
│  │                                                                   │  │
│  │ REVERSE EDGES (for bidirectional message passing):               │  │
│  │   • statute ──[rev_cites]──▶ case (1,655 edges)                 │  │
│  │   • chamber ──[rev_belongs_to]──▶ case (699 edges)              │  │
│  │   • case_type ──[rev_has_type]──▶ case (699 edges)              │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                     │                                                   │
│  Data Split:                                                            │
│  • Node split (outcome): train=491, val=69, test=139                   │
│  • Edge split (statute): train=1,159, val=165, test=331               │
│                     │                                                   │
│                     ▼                                                   │
│           data/graph_berturk_legal_new.pt                               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         GNN TRAINING                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  MODELS:                                                                │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ GAT (Graph Attention Network)                                    │  │
│  │   • GATv2Conv with multi-head attention                         │  │
│  │   • Learns attention weights for each neighbor                  │  │
│  │   • Separate convolution per edge type via HeteroConv           │  │
│  │                                                                   │  │
│  │ HAN (Hierarchical Attention Network)                            │  │
│  │   • HANConv with semantic attention                             │  │
│  │   • Two-level attention: node-level + meta-path-level          │  │
│  │   • Learns importance of different relationship types          │  │
│  │                                                                   │  │
│  │ HGT (Heterogeneous Graph Transformer) [BEST PERFORMER]          │  │
│  │   • HGTConv with type-specific query/key/value projections     │  │
│  │   • Relative temporal encoding via node/edge type              │  │
│  │   • Most powerful for complex heterogeneous graphs             │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  TASKS:                                                                 │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ 1. Outcome Prediction (Node Classification)                      │  │
│  │    Input: Case embedding from GNN                                │  │
│  │    Output: ONAMA (1) or BOZMA (0)                               │  │
│  │    Loss: Cross-entropy                                           │  │
│  │    Early stopping: Patience=10 on validation F1                 │  │
│  │                                                                   │  │
│  │ 2. Statute Recommendation (Link Prediction)                      │  │
│  │    Input: Case embedding + Statute embedding                    │  │
│  │    Output: Probability of citation (sigmoid)                    │  │
│  │    Loss: Binary cross-entropy with negative sampling            │  │
│  │    Evaluation: Precision@K, Recall@K, MRR                       │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  CHECKPOINTS (saved automatically):                                     │
│  • checkpoints/gat_berturk-legal_outcome.pt                            │
│  • checkpoints/gat_berturk-legal_statute.pt                            │
│  • checkpoints/han_berturk-legal_outcome.pt                            │
│  • checkpoints/han_berturk-legal_statute.pt                            │
│  • checkpoints/hgt_berturk-legal_outcome.pt                            │
│  • checkpoints/hgt_berturk-legal_statute.pt                            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🧠 Understanding the Graph Structure

### Why a Heterogeneous Graph?

A heterogeneous graph contains multiple types of nodes and edges. This is essential for legal analysis because:

1. **Cases** are the primary entities we want to analyze
2. **Statutes** represent the legal knowledge base
3. **Chambers** capture court specialization (labor law, commercial law, etc.)
4. **Case Types** enable category-based message passing

### Edge Semantics

#### 1. Citation Edges (case → statute)

When a court decision references a specific law article, we create a citation edge:

```
Case: "İş akdinin haksız feshi..."
   │
   └──[cites, weight=2.8]──▶ 4857-25 (İş Kanunu, haklı fesih nedenleri)
   └──[cites, weight=3.1]──▶ 4857-17 (İş Kanunu, ihbar süresi)
```

**TF-IDF Weighting**: Rare statutes get higher edge weights because they carry more information.

```python
weight = log(Total_Cases / Cases_Citing_This_Statute)
```

#### 2. Similarity Edges (case ↔ case)

Cases with similar plaintiff arguments are connected:

```
Case A: "Kıdem tazminatı ve ihbar tazminatı talep edilmektedir"
   │
   └──[similar, weight=0.95]──▶ Case B: "Kıdem ve ihbar tazminat alacağı"
```

This enables **information propagation**: If similar cases cited certain statutes, those are likely relevant.

#### 3. Chamber Edges (case → chamber)

Each case is heard by a specific court division:

```
Case (Labor dispute) ──[belongs_to]──▶ 9. Hukuk Dairesi
Case (Property dispute) ──[belongs_to]──▶ 20. Hukuk Dairesi
```

#### 4. Case Type Edges (case → case_type)

Each case has a classified type from LLM parsing:

```
Case (Wrongful termination) ──[has_type]──▶ ISE_IADE
Case (Compensation claim) ──[has_type]──▶ ALACAK_TAZMINAT
```

This enables a meta-path: Case → CaseType → Case, allowing similar case types to share information.

#### 5. Co-citation Edges (statute ↔ statute)

Statutes frequently cited together are connected:

```
4857-18 ◀──[co_cited]──▶ 4857-19 ◀──[co_cited]──▶ 4857-20
```

---

## 🔄 Message Passing: How GNNs Learn

### The Core Idea

GNNs work by propagating information between connected nodes. After several rounds of message passing, each node's representation contains information from its neighborhood.

### Meta-Paths

HAN and HGT consider different relationship paths:

| Meta-Path | Semantic |
|-----------|----------|
| Case → Statute → Case | Cases citing the same statutes |
| Case → Chamber → Case | Cases in the same court division |
| Case → CaseType → Case | Cases of the same category |
| Case → Similar → Case | Cases with similar plaintiff arguments |

---

## 📁 Project Structure

```
.
├── data/
│   ├── raw/                              # Raw scraped court decisions
│   │   └── decision_*.json               # Individual decision files
│   ├── parsed_all.json                   # Parsed with GPT-4o-mini (2,481 cases)
│   ├── embeddings_berturk_legal_new.pt   # Cached BERT embeddings
│   └── graph_berturk_legal_new.pt        # PyTorch Geometric graph
│
├── src/
│   ├── parser.py                         # 🔥 Main parsing module
│   │   ├── PROMPT_BASE                  # Unified extraction prompt
│   │   ├── parse_with_openai()          # GPT-4o-mini extraction with retry
│   │   ├── parse_openai_response()      # Response parsing
│   │   └── is_error_document()          # Error filtering
│   │
│   ├── scrape/                           # Data collection
│   │   └── scrape_api.py                # Yargıtay API scraper
│   │
│   ├── graph/                            # 🌟 GNN Module
│   │   ├── __init__.py                  # Public API exports
│   │   ├── embeddings.py                # Multi-model embedding generation
│   │   ├── builder.py                   # Heterogeneous graph construction
│   │   ├── models.py                    # GNN architectures (GAT, HAN, HGT)
│   │   └── trainer.py                   # Training, evaluation, checkpoint saving
│   │
│   ├── app.py                            # 🖥️ Streamlit UI for predictions
│   ├── predict_single.py                 # Single case prediction script
│   └── schema.py                         # Data classes (ParsedDecision, etc.)
│
├── checkpoints/                          # Saved model checkpoints
│   ├── gat_berturk-legal_outcome.pt
│   ├── gat_berturk-legal_statute.pt
│   ├── han_berturk-legal_outcome.pt
│   ├── han_berturk-legal_statute.pt
│   ├── hgt_berturk-legal_outcome.pt
│   └── hgt_berturk-legal_statute.pt
│
├── results/
│   ├── visualizations/                   # Generated plots
│   │   ├── node_types.png
│   │   ├── edge_types.png
│   │   ├── outcome_dist.png
│   │   └── similarity_weights.png
│   └── report_*.md                       # Experiment reports
│
├── analyze_pipeline.py                   # Full pipeline runner
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone <repository-url>
cd legal-graph-analysis

# Create conda environment
conda create -n graph-project python=3.10
conda activate graph-project

# Install dependencies
pip install -r requirements.txt

# Install PyTorch Geometric
pip install torch-geometric
```

### 2. Set API Keys

```bash
# Create .env file
echo "OPENAI_API_KEY=your-key-here" > .env
```

### 3. Run Full Pipeline

```bash
# Step 1: Parse raw documents with GPT-4o-mini
python -m src.parser --openai --output data/parsed_all.json --workers 5

# Step 2: Run analysis pipeline
python analyze_pipeline.py --parsed-file data/parsed_all.json --embedding berturk-legal

# Step 3: Save results to separate directory
mv results results_berturk_legal
```

### 4. Run the UI

```bash
streamlit run src/app.py
```

Open http://localhost:8501 in your browser. The UI allows you to:
- Input petition text
- Select model type (GAT, HAN, HGT)
- Get outcome prediction (ONAMA/BOZMA with confidence)
- Get top-K statute recommendations

---

## 🔧 Detailed Usage

### Parsing

The parser extracts structured information from raw court decisions using GPT-4o-mini:

```bash
# Parse with GPT-4o-mini (with automatic retry for rate limits)
python -m src.parser --openai --output data/parsed.json --workers 5

# Limit files for testing
python -m src.parser --openai --limit 100 --output data/test.json
```

**Parsed Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique decision ID |
| `plaintiff_arguments` | string | Davacı/vekili iddia ve talepleri |
| `case_type_enum` | enum | ISE_IADE, ALACAK_TAZMINAT, IS_KAZASI, etc. |
| `outcome` | enum | ONAMA, BOZMA, GOREVSIZLIK, GERI_CEVIRME |
| `chamber` | string | "9. Hukuk Dairesi", "21. Hukuk Dairesi", etc. |
| `statute_ids` | list | ["5521-1", "4857-18", "6100-22", ...] |

### Analyze Pipeline

```bash
# Standard run with BERTurk-Legal embeddings
python analyze_pipeline.py --parsed-file data/parsed_all.json --embedding berturk-legal

# With OpenAI embeddings
python analyze_pipeline.py --parsed-file data/parsed_all.json --embedding openai

# Skip training (just analyze data and build graph)
python analyze_pipeline.py --parsed-file data/parsed.json --skip-training

# Adjust similarity edges (top-K neighbors per case)
python analyze_pipeline.py --parsed-file data/parsed.json --similarity-k 15
```

### Single Prediction

```bash
python src/predict_single.py \
    --graph data/graph_berturk_legal_new.pt \
    --ckpt-outcome checkpoints/hgt_berturk-legal_outcome.pt \
    --ckpt-statute checkpoints/hgt_berturk-legal_statute.pt \
    --model hgt \
    --chamber "9. Hukuk Dairesi" \
    --text "Davacı vekili; müvekkilinin iş akdinin haksız feshedildiğini..." \
    --topk 10 \
    --json
```

### API Usage

```python
from src.graph import generate_embeddings, LegalGraphBuilder, create_model
from src.graph.trainer import train_outcome_model, train_statute_model

# 1. Generate embeddings
embeddings = generate_embeddings(
    texts=["Davacı, kıdem tazminatı talep etmektedir..."],
    model="berturk-legal",
    cache_file="embeddings.pt"
)

# 2. Build graph
builder = LegalGraphBuilder("data/parsed.json", embeddings=embeddings)
data = builder.build(similarity_k=10, similarity_threshold=0.3)
data = builder.create_train_test_split(data)

# 3. Train model
model = create_model("hgt", data)
outcome_results = train_outcome_model(model, data, num_epochs=100)
print(f"F1: {outcome_results['f1']:.4f}")
```

---

## 📈 Understanding the Metrics

### Outcome Prediction Metrics

| Metric | Description |
|--------|-------------|
| **Accuracy** | Percentage of correctly predicted cases |
| **F1 Score** | Harmonic mean of precision and recall |
| **Precision** | Of predicted BOZMA, how many are correct |
| **Recall** | Of actual BOZMA, how many did we find |

### Statute Recommendation Metrics

| Metric | Description |
|--------|-------------|
| **Precision@K** | Of top-K recommendations, how many are actually cited |
| **Recall@K** | Of all true statutes, how many appear in top-K |
| **MRR** | Mean Reciprocal Rank of first correct statute |

---

## 🔬 Technical Details

### Extraction Prompt

The LLM prompt for parsing is designed for Turkish legal documents:

```
ALAN ÇIKARIMI (ÖNEMLİ!):

1. plaintiff_arguments: Sadece davacı/vekili iddiaları
2. case_type_enum: ISE_IADE | ALACAK_TAZMINAT | IS_KAZASI | 
                   HIZMET_TESPITI | GOREV_UYUSMAZLIGI
3. outcome: ONAMA | BOZMA | GOREVSIZLIK | GERI_CEVIRME
4. statute_ids: ["4857-25", "6100-22", ...]

KANUN MADDELERİ:
• HUMK → 1086, HMK → 6100, İşK → 4857, BK → 818
• "4857-18" formatında döndür
```

### GNN Hyperparameters

| Parameter | Value |
|-----------|-------|
| Hidden dimension | 128 |
| Number of heads | 4 |
| Number of layers | 2 |
| Dropout | 0.3 |
| Learning rate | 0.01 |
| Weight decay | 1e-4 |
| Early stopping patience | 10 |

### Train/Val/Test Split

- **Node split** (outcome prediction): 70% / 10% / 20%
- **Edge split** (statute recommendation): 70% / 10% / 20%

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

MIT License

## 📚 Citation

```bibtex
@misc{yargitay-gnn-2024,
  title={Legal Graph Analysis: Statute Recommendation and Outcome Prediction using Graph Neural Networks},
  author={YZV413 Graph Theory Term Project},
  year={2024},
  institution={Istanbul Technical University}
}
```

---

## 🙏 Acknowledgments

- **KocLab-Bilkent** for the BERTurk-Legal model
- **PyTorch Geometric** team for the excellent GNN library
- **Yargıtay** for providing public access to court decisions

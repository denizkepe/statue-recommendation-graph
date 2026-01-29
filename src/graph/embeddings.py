"""
Multi-Embedding Support for Legal Graph.

Supports three embedding models:
1. BERTurk (dbmdz/bert-base-turkish-cased) - General Turkish
2. BERTurk-Legal (KocLab-Bilkent/BERTurk-Legal) - Legal domain
3. OpenAI (text-embedding-3-small) - Best quality

Usage:
    embeddings = generate_embeddings(texts, model="berturk-legal")
"""

import os
import torch
import numpy as np
from pathlib import Path
from typing import List, Literal, Optional
from tqdm import tqdm

# Embedding model configurations
EMBEDDING_MODELS = {
    "berturk": {
        "name": "dbmdz/bert-base-turkish-cased",
        "dim": 768,
        "description": "General Turkish BERT",
    },
    "berturk-legal": {
        "name": "KocLab-Bilkent/BERTurk-Legal",
        "dim": 768,
        "description": "Legal domain Turkish BERT (recommended)",
    },
    "openai": {
        "name": "text-embedding-3-small",
        "dim": 1536,
        "description": "OpenAI embeddings (best quality, requires API key)",
    },
}


def generate_bert_embeddings(
    texts: List[str],
    model_name: str,
    batch_size: int = 8,
    max_length: int = 512,
) -> torch.Tensor:
    """
    Generate embeddings using a BERT model.
    
    Args:
        texts: List of text strings
        model_name: HuggingFace model name
        batch_size: Batch size for inference
        max_length: Maximum sequence length
        
    Returns:
        Tensor of shape [num_texts, hidden_dim]
    """
    from transformers import AutoTokenizer, AutoModel
    
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    
    embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
        batch = texts[i:i + batch_size]
        # Handle empty texts
        batch = [t[:max_length] if t else "[PAD]" for t in batch]
        
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        
        with torch.no_grad():
            outputs = model(**inputs)
            # Use [CLS] token embedding
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            embeddings.append(cls_embeddings)
    
    return torch.cat(embeddings, dim=0)


def generate_openai_embeddings(
    texts: List[str],
    model_name: str = "text-embedding-3-small",
    batch_size: int = 100,
) -> torch.Tensor:
    """
    Generate embeddings using OpenAI API.
    
    Requires OPENAI_API_KEY environment variable.
    
    Args:
        texts: List of text strings
        model_name: OpenAI model name
        batch_size: Batch size for API calls
        
    Returns:
        Tensor of shape [num_texts, 1536]
    """
    from openai import OpenAI
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    client = OpenAI(api_key=api_key)
    embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating OpenAI embeddings"):
        batch = texts[i:i + batch_size]
        batch = [t[:8000] if t else "empty" for t in batch]
        
        response = client.embeddings.create(
            model=model_name,
            input=batch,
        )
        
        for item in response.data:
            embeddings.append(item.embedding)
    
    return torch.tensor(embeddings, dtype=torch.float32)


def generate_embeddings(
    texts: List[str],
    model: Literal["berturk", "berturk-legal", "openai"] = "berturk-legal",
    cache_file: Optional[str] = None,
    force_regenerate: bool = False,
) -> torch.Tensor:
    """
    Generate embeddings using specified model.
    
    Args:
        texts: List of text strings to embed
        model: One of "berturk", "berturk-legal", "openai"
        cache_file: Path to cache embeddings (optional)
        force_regenerate: If True, regenerate even if cache exists
        
    Returns:
        Tensor of embeddings
        
    Example:
        embeddings = generate_embeddings(
            texts=["Davacı kıdem tazminatı talep etmiştir"],
            model="berturk-legal",
            cache_file="data/embeddings_legal.pt"
        )
    """
    # Check cache
    if cache_file and Path(cache_file).exists() and not force_regenerate:
        print(f"Loading cached embeddings from {cache_file}")
        return torch.load(cache_file, weights_only=False)
    
    # Get model config
    if model not in EMBEDDING_MODELS:
        raise ValueError(f"Unknown model: {model}. Choose from {list(EMBEDDING_MODELS.keys())}")
    
    config = EMBEDDING_MODELS[model]
    print(f"Using {config['description']} ({config['name']})")
    
    # Generate embeddings
    if model == "openai":
        embeddings = generate_openai_embeddings(texts, config["name"])
    else:
        embeddings = generate_bert_embeddings(texts, config["name"])
    
    # Cache
    if cache_file:
        Path(cache_file).parent.mkdir(parents=True, exist_ok=True)
        torch.save(embeddings, cache_file)
        print(f"Saved embeddings to {cache_file}")
    
    return embeddings


def compare_embeddings(
    texts: List[str],
    models: List[str] = ["berturk", "berturk-legal", "openai"],
    cache_dir: str = "data/embeddings",
) -> dict:
    """
    Generate embeddings with multiple models for comparison.
    
    Args:
        texts: List of text strings
        models: Models to compare
        cache_dir: Directory to cache embeddings
        
    Returns:
        Dictionary of model_name -> embeddings tensor
    """
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    for model in models:
        cache_file = Path(cache_dir) / f"embeddings_{model.replace('-', '_')}.pt"
        try:
            embeddings = generate_embeddings(
                texts,
                model=model,
                cache_file=str(cache_file),
            )
            results[model] = embeddings
            print(f"  {model}: {embeddings.shape}")
        except Exception as e:
            print(f"  {model}: FAILED - {e}")
    
    return results


def get_embedding_dim(model: str) -> int:
    """Get embedding dimension for a model."""
    return EMBEDDING_MODELS[model]["dim"]


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate embeddings for legal texts")
    parser.add_argument("--input", required=True, help="JSON file with cases")
    parser.add_argument("--model", default="berturk-legal", choices=EMBEDDING_MODELS.keys())
    parser.add_argument("--output", default="data/embeddings.pt")
    parser.add_argument("--field", default="plaintiff_arguments", help="Text field to embed")
    args = parser.parse_args()
    
    import json
    
    with open(args.input, "r", encoding="utf-8") as f:
        cases = json.load(f)
    
    texts = [c.get(args.field, "") for c in cases]
    print(f"Loaded {len(texts)} texts from {args.input}")
    
    embeddings = generate_embeddings(texts, model=args.model, cache_file=args.output)
    print(f"Generated embeddings: {embeddings.shape}")

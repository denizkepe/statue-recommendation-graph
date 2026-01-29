# scripts/predict_single.py
import sys
import json
from pathlib import Path

# repo root = .../yargitay-hukuk-graph
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from transformers import AutoTokenizer, AutoModel

from src.graph.models import create_model  # sendeki yol bu


def load_ckpt_state_dict(path: str):
    """
    Bizim save_checkpoint formatı genelde:
      {"state_dict": ..., "meta": ...}
    ama bazen direkt state_dict de olabilir.
    """
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"]
    if isinstance(obj, dict) and all(isinstance(k, str) for k in obj.keys()):
        return obj
    raise ValueError(f"Checkpoint formatı beklenmedik: {path}")


def embed_berturk_legal(text: str, device="cpu") -> torch.Tensor:
    model_name = "KocLab-Bilkent/BERTurk-Legal"
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name).to(device)
    mdl.eval()

    t = (text or "").strip()
    if not t:
        t = "[BOS]"

    with torch.no_grad():
        inputs = tok([t[:512]], return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        out = mdl(**inputs)
        cls = out.last_hidden_state[:, 0, :]  # [1,768]
    return cls.cpu()


def _append_edge(data: HeteroData, edge_type, src_idx: int, dst_idx: int):
    new_col = torch.tensor([[src_idx], [dst_idx]], dtype=torch.long)
    ei = data[edge_type].edge_index
    data[edge_type].edge_index = torch.cat([ei, new_col], dim=1)


def add_query_case_node(data: HeteroData, x_new: torch.Tensor, chamber: str | None):
    """
    Query case node'u graf'a ekler:
      - case.x append
      - case->chamber ve chamber->case reverse edge ekler (varsa)
      - label/mask alanlarını uzatır (unlabeled)
    """
    n_old = data["case"].x.size(0)

    # 1) case feature append
    data["case"].x = torch.cat([data["case"].x, x_new], dim=0)

    # node_ids
    if hasattr(data["case"], "node_ids"):
        data["case"].node_ids = list(data["case"].node_ids) + ["QUERY_CASE"]

    # y + labeled_mask uzat (unlabeled)
    if hasattr(data["case"], "y"):
        y_old = data["case"].y
        data["case"].y = torch.cat([y_old, torch.tensor([-1], dtype=y_old.dtype)], dim=0)

    if hasattr(data["case"], "labeled_mask"):
        lm_old = data["case"].labeled_mask
        data["case"].labeled_mask = torch.cat([lm_old, torch.tensor([False], dtype=lm_old.dtype)], dim=0)

    # split maskleri uzat
    for m in ["train_mask", "val_mask", "test_mask"]:
        if hasattr(data["case"], m):
            old = getattr(data["case"], m)
            setattr(data["case"], m, torch.cat([old, torch.tensor([False], dtype=old.dtype)], dim=0))

    # 2) connect to chamber (forward + reverse)
    if chamber is not None:
        if not hasattr(data["chamber"], "node_ids"):
            raise ValueError("Graph içinde chamber.node_ids yok. Builder bunu yazmıyor olabilir.")

        chamber_ids = list(data["chamber"].node_ids)
        if chamber not in chamber_ids:
            raise ValueError(f"Unknown chamber: {chamber}. Available sample: {chamber_ids[:10]}")

        ch_idx = chamber_ids.index(chamber)

        fwd = ("case", "belongs_to", "chamber")
        rev = ("chamber", "rev_belongs_to", "case")

        if fwd in data.edge_types:
            _append_edge(data, fwd, n_old, ch_idx)
        else:
            raise ValueError(f"Graph edge type missing: {fwd}")

        if rev in data.edge_types:
            _append_edge(data, rev, ch_idx, n_old)

    return n_old  # query idx


def split_statute_id(sid: str):
    """
    "6100-22" -> ("6100", "22")
    "4857-21/3" -> ("4857", "21/3")
    "5510" -> ("5510", None)
    """
    sid = (sid or "").strip()
    if "-" not in sid:
        return sid, None
    law_no, article = sid.split("-", 1)
    return law_no.strip(), article.strip() if article.strip() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="data/graph_berturk_legal.pt")
    ap.add_argument("--ckpt-outcome", required=True)
    ap.add_argument("--ckpt-statute", required=True)
    ap.add_argument("--model", choices=["gat", "han", "hgt"], default="hgt")
    ap.add_argument("--text", required=True)
    ap.add_argument("--chamber", default=None, help='e.g. "9. Hukuk Dairesi"')
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--json", action="store_true", help="print machine-readable JSON only")
    args = ap.parse_args()

    data: HeteroData = torch.load(args.graph, map_location="cpu", weights_only=False)

    # Query embed
    x_new = embed_berturk_legal(args.text)  # [1,768]
    q_idx = add_query_case_node(data, x_new, chamber=args.chamber)

    # --- OUTCOME MODEL ---
    outcome_model = create_model(args.model, data)
    outcome_sd = load_ckpt_state_dict(args.ckpt_outcome)
    outcome_model.load_state_dict(outcome_sd, strict=False)
    outcome_model.eval()

    # --- STATUTE MODEL ---
    statute_model = create_model(args.model, data)
    statute_sd = load_ckpt_state_dict(args.ckpt_statute)
    statute_model.load_state_dict(statute_sd, strict=False)
    statute_model.eval()

    with torch.no_grad():
        # OUTCOME
        emb_o = outcome_model(data)
        logits = outcome_model.predict_outcome(emb_o)
        if isinstance(logits, dict):
            logits = logits["case"]
        prob = F.softmax(logits[q_idx], dim=0)
        p_bozma = float(prob[0].item())
        p_onama = float(prob[1].item())
        outcome = "ONAMA" if int(prob.argmax().item()) == 1 else "BOZMA"

        # STATUTE
        emb_s = statute_model(data)
        scores = statute_model.recommend_statutes(emb_s)
        if isinstance(scores, dict):
            scores = scores["case"]

        s = scores[q_idx]  # [num_statute]
        k = min(args.topk, s.numel())
        top_idx = torch.topk(s, k=k).indices.tolist()

    statute_ids = list(data["statute"].node_ids)
    recs = [statute_ids[i] for i in top_idx]
    rec_rows = []
    for rank, sid in enumerate(recs, start=1):
        law_no, article = split_statute_id(sid)
        rec_rows.append({"rank": rank, "statute_id": sid, "law_no": law_no, "article": article})

    payload = {
        "model": args.model,
        "chamber": args.chamber,
        "topk": args.topk,
        "outcome": {"pred": outcome, "p_bozma": p_bozma, "p_onama": p_onama},
        "recommendations": rec_rows,
    }

    if args.json:
        print(json.dumps(payload, ensure_ascii=False))
        return

    print("\n=== OUTCOME ===")
    print("pred:", outcome, "| p(BOZMA)=", p_bozma, "p(ONAMA)=", p_onama)
    print("\n=== STATUTE RECOMMENDATION ===")
    for r in recs:
        print("-", r)


if __name__ == "__main__":
    main()

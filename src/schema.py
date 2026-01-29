"""
Simple Data Schema for Legal Graph.

Contains the essential fields for graph-based prediction plus
useful metadata for meta-paths and analysis.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Case:
    """
    A parsed court case with fields needed for GNN.
    
    Core fields:
        - id: Unique identifier
        - claim_summary: Davacı istemi (input for prediction)
        - statute_ids: Cited laws (prediction target)
    
    Useful metadata:
        - outcome: BOZMA/ONAMA (secondary classification)
        - chamber: e.g., "9. Hukuk Dairesi" (for Case-Chamber-Case meta-path)
        - year: Decision year (temporal analysis)
    """
    
    # Core fields (required for graph)
    id: str
    claim_summary: str
    statute_ids: List[str] = field(default_factory=list)
    
    # Useful metadata (for meta-paths and analysis)
    outcome: Optional[str] = None      # BOZMA, ONAMA, PARTIAL
    chamber: Optional[str] = None      # 9. Hukuk Dairesi, etc.
    year: Optional[int] = None
    
    # Optional detailed fields
    plaintiff_arguments: Optional[str] = None
    
    def to_dict(self):
        return {
            "id": self.id,
            "claim_summary": self.claim_summary,
            "statute_ids": self.statute_ids,
            "outcome": self.outcome,
            "chamber": self.chamber,
            "year": self.year,
            "plaintiff_arguments": self.plaintiff_arguments,
        }
    
    @classmethod
    def from_dict(cls, d: dict):
        return cls(
            id=d.get("id", ""),
            claim_summary=d.get("claim_summary", ""),
            statute_ids=d.get("statute_ids", []),
            outcome=d.get("outcome") or d.get("outcome_class"),
            chamber=d.get("chamber"),
            year=d.get("year") or (int(d["decision_date"][:4]) if d.get("decision_date") else None),
            plaintiff_arguments=d.get("plaintiff_arguments"),
        )


# Known Turkish Laws
KNOWN_LAWS = {
    "4857": "İş Kanunu",
    "6100": "Hukuk Muhakemeleri Kanunu",
    "1086": "Hukuk Usulü Muhakemeleri Kanunu",
    "5510": "Sosyal Sigortalar ve Genel Sağlık Sigortası Kanunu",
    "506": "Sosyal Sigortalar Kanunu",
    "6098": "Türk Borçlar Kanunu",
    "818": "Borçlar Kanunu",
    "4721": "Türk Medeni Kanunu",
    "5521": "İş Mahkemeleri Kanunu",
}

# Abbreviation mappings
ABBREVIATION_MAP = {
    "HMK": "6100",
    "HUMK": "1086",
    "BK": "818",
    "TBK": "6098",
    "İşK": "4857",
    "SSK": "506",
    "SGK": "5510",
    "TMK": "4721",
}

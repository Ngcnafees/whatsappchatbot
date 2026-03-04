import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import faiss
import numpy as np


class ImageSimilarityIndex:
    """
    Loads:
      - index.faiss (IndexFlatIP / cosine)
      - meta.jsonl  (row i -> metadata dict)
    Allows:
      - similar_by_stockcode(code, k=3) -> list of meta rows
    """

    def __init__(self, index_dir: str):
        self.index_dir = Path(index_dir)
        self.index = faiss.read_index(str(self.index_dir / "index.faiss"))
        self.meta = self._load_meta(self.index_dir / "meta.jsonl")

        # Build stockcode -> row_id map
        self.code_to_id: Dict[str, int] = {}
        for i, row in enumerate(self.meta):
            sc = self._norm(row.get("stockcode", ""))
            if sc and sc not in self.code_to_id:
                self.code_to_id[sc] = i

    def _load_meta(self, path: Path) -> List[Dict[str, Any]]:
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def _norm(self, s: str) -> str:
        return (s or "").strip().upper()

    def _vector_by_id(self, idx: int) -> np.ndarray:
        """
        Reconstruct stored vector from FAISS.
        Works with IndexFlatIP / IndexFlatL2.
        """
        v = self.index.reconstruct(idx)  # shape (dim,)
        return np.asarray(v, dtype="float32")

    def similar_by_stockcode(self, stockcode: str, k: int = 3) -> List[Dict[str, Any]]:
        code = self._norm(stockcode)
        if code not in self.code_to_id:
            return []

        base_id = self.code_to_id[code]
        q = self._vector_by_id(base_id)
        q = np.expand_dims(q, axis=0)  # (1, dim)

        # ask for more than k because the first will usually be itself
        scores, ids = self.index.search(q, k + 6)

        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            if idx == base_id:
                continue  # skip itself
            row = dict(self.meta[idx])
            row["score"] = float(score)
            results.append(row)
            if len(results) >= k:
                break

        return results

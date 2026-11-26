"""Avalia modelos com a mesma divisão e as mesmas métricas."""
import pandas as pd

from .metrics import ndcg_at_k, precision_at_k, recall_at_k
from .models.base import Recommender


def evaluate(
    model: Recommender, train: pd.DataFrame, test: pd.DataFrame,
    k: int = 10, threshold: float = 4.0,
) -> dict:
    relevant = (
        test[test["rating"] >= threshold].groupby("userId")["movieId"].apply(set)
    )
    rows, recommended = [], set()
    for user_id, rel in relevant.items():
        recs = model.recommend(user_id, k)
        recommended.update(recs)
        rows.append((precision_at_k(recs, rel, k), recall_at_k(recs, rel, k), ndcg_at_k(recs, rel, k)))
    p, r, n = pd.DataFrame(rows).mean()
    return {
        "modelo": model.name,
        f"precision@{k}": round(p, 4),
        f"recall@{k}": round(r, 4),
        f"ndcg@{k}": round(n, 4),
        "cobertura": round(len(recommended) / train["movieId"].nunique(), 4),
        "usuarios": len(relevant),
    }

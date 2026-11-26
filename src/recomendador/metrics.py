"""Métricas de ranking para recomendação top-K."""
import math
from collections.abc import Sequence


def precision_at_k(recs: Sequence[int], relevant: set[int], k: int) -> float:
    if k == 0:
        return 0.0
    return sum(1 for item in recs[:k] if item in relevant) / k


def recall_at_k(recs: Sequence[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    return sum(1 for item in recs[:k] if item in relevant) / len(relevant)


def ndcg_at_k(recs: Sequence[int], relevant: set[int], k: int) -> float:
    """Premia acertos nas primeiras posições do ranking."""
    dcg = sum(1 / math.log2(i + 2) for i, item in enumerate(recs[:k]) if item in relevant)
    idcg = sum(1 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg else 0.0

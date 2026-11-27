"""Avalia modelos com a mesma divisão e as mesmas métricas.

evaluate_per_user() guarda as métricas de cada usuário, o que permite calcular
intervalos de confiança e comparar modelos usuário a usuário (ver stats.py).
"""
import pandas as pd

from .metrics import ndcg_at_k, precision_at_k, recall_at_k
from .models.base import Recommender

METRICAS = ("precision", "recall", "ndcg")


def evaluate_per_user(
    model: Recommender, test: pd.DataFrame, k: int = 10, threshold: float = 4.0
) -> pd.DataFrame:
    """Uma linha por usuário: precision, recall, ndcg e a lista recomendada."""
    relevant = (
        test[test["rating"] >= threshold].groupby("userId")["movieId"].apply(set)
    )
    rows = []
    for user_id, rel in relevant.items():
        recs = model.recommend(user_id, k)
        rows.append({
            "userId": user_id,
            "precision": precision_at_k(recs, rel, k),
            "recall": recall_at_k(recs, rel, k),
            "ndcg": ndcg_at_k(recs, rel, k),
            "recs": recs,
        })
    return pd.DataFrame(rows).set_index("userId")


def summarize(per_user: pd.DataFrame, name: str, n_items_train: int, k: int = 10) -> dict:
    recommended = {item for recs in per_user["recs"] for item in recs}
    resumo = {"modelo": name}
    for m in METRICAS:
        resumo[f"{m}@{k}"] = round(per_user[m].mean(), 4)
    resumo["cobertura"] = round(len(recommended) / n_items_train, 4)
    resumo["usuarios"] = len(per_user)
    return resumo


def evaluate(
    model: Recommender, train: pd.DataFrame, test: pd.DataFrame,
    k: int = 10, threshold: float = 4.0,
) -> dict:
    per_user = evaluate_per_user(model, test, k, threshold)
    return summarize(per_user, model.name, train["movieId"].nunique(), k)

"""Divisão treino/teste temporal por usuário.

Para cada usuário, as últimas N avaliações vão para teste. Isso simula o uso
real (prever o que a pessoa vai assistir depois) melhor que um split aleatório.
"""
import pandas as pd


def temporal_leave_last_out(
    ratings: pd.DataFrame, n_test: int = 5, min_ratings: int = 10
) -> tuple[pd.DataFrame, pd.DataFrame]:
    r = ratings.sort_values(["userId", "timestamp"])
    counts = r.groupby("userId")["movieId"].transform("size")
    r = r[counts >= min_ratings]
    pos_from_end = r.groupby("userId").cumcount(ascending=False)
    test = r[pos_from_end < n_test]
    train = r[pos_from_end >= n_test]
    return train.reset_index(drop=True), test.reset_index(drop=True)

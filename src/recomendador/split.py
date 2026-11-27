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


def remove_test_tags(tags: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Remove tags que um usuário aplicou a filmes que estão no teste dele.

    Sem isso, o modelo de conteúdo "veria" a opinião do usuário sobre o filme
    que ele deveria adivinhar (vazamento de dados).
    """
    pares_teste = test[["userId", "movieId"]].drop_duplicates().assign(_teste=True)
    merged = tags.merge(pares_teste, on=["userId", "movieId"], how="left")
    return merged[merged["_teste"].isna()].drop(columns="_teste").reset_index(drop=True)

"""Baselines: aleatório e popularidade.

Todo modelo sofisticado precisa vencer esses dois para justificar a complexidade.
"""
import numpy as np
import pandas as pd

from .base import Recommender


class RandomRecommender(Recommender):
    name = "Aleatório"

    def __init__(self, seed: int = 42):
        self.seed = seed

    def fit(self, ratings: pd.DataFrame) -> "RandomRecommender":
        self._remember_seen(ratings)
        self._items = ratings["movieId"].unique()
        self._rng = np.random.default_rng(self.seed)
        return self

    def _score(self, user_id: int) -> pd.Series:
        return pd.Series(self._rng.random(len(self._items)), index=self._items)


class PopularityRecommender(Recommender):
    """criterio='contagem': filmes mais avaliados.
    criterio='media_ponderada': média bayesiana (fórmula do IMDb), que evita
    que um filme com uma única nota 5 apareça no topo.
    """

    def __init__(self, criterio: str = "contagem", quantil_votos: float = 0.9):
        if criterio not in {"contagem", "media_ponderada"}:
            raise ValueError("criterio deve ser 'contagem' ou 'media_ponderada'")
        self.criterio = criterio
        self.quantil_votos = quantil_votos
        self.name = f"Popularidade ({criterio.replace('_', ' ')})"

    def fit(self, ratings: pd.DataFrame) -> "PopularityRecommender":
        self._remember_seen(ratings)
        stats = ratings.groupby("movieId")["rating"].agg(["mean", "count"])
        if self.criterio == "contagem":
            self.scores_ = stats["count"].astype(float)
        else:
            v, r = stats["count"], stats["mean"]
            m = v.quantile(self.quantil_votos)
            c = ratings["rating"].mean()
            self.scores_ = (v / (v + m)) * r + (m / (v + m)) * c
        return self

    def _score(self, user_id: int) -> pd.Series:
        return self.scores_

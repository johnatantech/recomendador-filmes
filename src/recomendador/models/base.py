"""Interface comum: todo modelo implementa fit() e _score().

Com a mesma interface, a avaliação trata todos os modelos igual e
adicionar uma abordagem nova não exige mudar o resto do código.
"""
from abc import ABC, abstractmethod

import pandas as pd


class Recommender(ABC):
    name: str = "base"

    def _remember_seen(self, ratings: pd.DataFrame) -> None:
        self._seen = ratings.groupby("userId")["movieId"].apply(set).to_dict()

    @abstractmethod
    def fit(self, ratings: pd.DataFrame) -> "Recommender": ...

    @abstractmethod
    def _score(self, user_id: int) -> pd.Series:
        """Série indexada por movieId com a pontuação de cada filme."""

    def recommend(self, user_id: int, k: int = 10, exclude_seen: bool = True) -> list[int]:
        scores = self._score(user_id)
        if exclude_seen:
            scores = scores.drop(list(self._seen.get(user_id, ())), errors="ignore")
        return scores.nlargest(k).index.tolist()

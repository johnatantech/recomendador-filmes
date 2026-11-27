"""Interface comum: todo modelo implementa fit() e _score().

Com a mesma interface, a avaliação trata todos os modelos igual e
adicionar uma abordagem nova não exige mudar o resto do código.

Modelos que implementam _score_history() também recomendam para um usuário
NOVO, que não estava no treino (é o que o app usa: a pessoa marca filmes de
que gosta e recebe recomendações na hora, sem retreinar o modelo).
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

    def _score_history(self, history: dict[int, float]) -> pd.Series:
        """Pontuação para um usuário novo, a partir de {movieId: nota}."""
        raise NotImplementedError(f"{self.name} não recomenda para usuários novos")

    @staticmethod
    def _top_k(scores: pd.Series, excluir, k: int) -> list[int]:
        return scores.drop(list(excluir), errors="ignore").nlargest(k).index.tolist()

    def recommend(self, user_id: int, k: int = 10, exclude_seen: bool = True) -> list[int]:
        excluir = self._seen.get(user_id, ()) if exclude_seen else ()
        return self._top_k(self._score(user_id), excluir, k)

    def recommend_for_history(self, history: dict[int, float], k: int = 10) -> list[int]:
        return self._top_k(self._score_history(history), history.keys(), k)

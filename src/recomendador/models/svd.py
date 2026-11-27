"""Fatoração de matrizes com SVD truncado (PureSVD).

A ideia: o gosto das pessoas tem poucas "dimensões escondidas" (fatores latentes),
como "gosta de ação dos anos 90" ou "prefere filmes cult". O SVD comprime a matriz
usuário × filme nessas F dimensões.

1. Monta a matriz esparsa usuário × filme (filmes não vistos = 0).
2. Decompõe com SVD truncado: X ≈ U · S · Vᵀ, mantendo só F fatores.
3. V (filmes × F) descreve cada filme no espaço de gostos.
4. A pontuação do usuário é a projeção do histórico dele nesse espaço: x · V · Vᵀ.

Referência: Cremonesi, Koren e Turrin (2010), "Performance of recommender
algorithms on top-N recommendation tasks", RecSys.
"""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

from .base import Recommender


class SVDRecommender(Recommender):
    def __init__(self, fatores: int = 50, implicito: bool = True, seed: int = 42):
        self.fatores = fatores
        self.implicito = implicito
        self.seed = seed
        tipo = "implícito" if implicito else "notas"
        self.name = f"SVD {fatores} fatores ({tipo})"

    def fit(self, ratings: pd.DataFrame) -> "SVDRecommender":
        self._remember_seen(ratings)
        self._items = np.sort(ratings["movieId"].unique())
        users = np.sort(ratings["userId"].unique())
        self._user_pos = pd.Series(np.arange(len(users)), index=users)
        item_pos = pd.Series(np.arange(len(self._items)), index=self._items)
        self._item_pos = item_pos

        valores = (
            np.ones(len(ratings)) if self.implicito else ratings["rating"].to_numpy(float)
        )
        self._X = csr_matrix(
            (valores, (self._user_pos.loc[ratings["userId"]], item_pos.loc[ratings["movieId"]])),
            shape=(len(users), len(self._items)),
        )

        menor = min(self._X.shape)
        k = min(self.fatores, menor - 1)
        # v0 fixo deixa o resultado reproduzível (o ARPACK parte de um vetor aleatório)
        v0 = np.random.default_rng(self.seed).random(menor)
        _, _, vt = svds(self._X, k=k, v0=v0)
        self.V_ = vt.T  # filmes × fatores
        return self

    def _score(self, user_id: int) -> pd.Series:
        if user_id not in self._user_pos.index:
            return pd.Series(0.0, index=self._items)
        historico = self._X[self._user_pos[user_id]]  # 1 × filmes
        scores = (historico @ self.V_) @ self.V_.T     # 1 × filmes
        return pd.Series(np.asarray(scores).ravel(), index=self._items)

    def _score_history(self, history: dict[int, float]) -> pd.Series:
        filmes = [m for m in history if m in self._item_pos.index]
        x = np.zeros(len(self._items))
        x[self._item_pos.loc[filmes].to_numpy()] = (
            1.0 if self.implicito else [history[m] for m in filmes]
        )
        return pd.Series((x @ self.V_) @ self.V_.T, index=self._items)

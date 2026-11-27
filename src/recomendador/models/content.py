"""Recomendação baseada em conteúdo com TF-IDF de gêneros e tags.

1. Cada filme vira um "documento" com seus gêneros e tags.
2. O TF-IDF transforma os documentos em vetores, dando mais peso a termos raros
   (ex.: "film-noir" diz mais sobre um filme que "drama").
3. O perfil do usuário é a média ponderada dos vetores dos filmes que ele avaliou,
   com peso = nota - média do usuário (gostou acima do normal pesa positivo,
   abaixo do normal pesa negativo).
4. A pontuação de cada filme é a similaridade entre o vetor dele e o perfil.
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .base import Recommender


class ContentRecommender(Recommender):
    def __init__(self, movies: pd.DataFrame, tags: pd.DataFrame | None = None):
        self.movies = movies
        self.tags = tags
        self.name = "Conteúdo (gêneros + tags)" if tags is not None else "Conteúdo (gêneros)"

    def _documents(self, item_ids: np.ndarray) -> pd.Series:
        genres = (
            self.movies.set_index("movieId")["genres"].reindex(item_ids).fillna("")
            .str.replace("(no genres listed)", "", regex=False)
            .str.replace("-", "", regex=False)  # "Sci-Fi" vira um único termo
            .str.replace("|", " ", regex=False)
            .str.lower()
        )
        if self.tags is None:
            return genres
        tag_docs = (
            self.tags.assign(tag=self.tags["tag"].astype(str).str.lower())
            .groupby("movieId")["tag"].apply(" ".join)
            .reindex(item_ids).fillna("")
        )
        return genres + " " + tag_docs

    def fit(self, ratings: pd.DataFrame) -> "ContentRecommender":
        self._remember_seen(ratings)
        self._items = np.sort(ratings["movieId"].unique())
        pos = pd.Series(np.arange(len(self._items)), index=self._items)

        self.vectorizer_ = TfidfVectorizer()
        self._X = self.vectorizer_.fit_transform(self._documents(self._items))

        r = ratings.assign(
            peso=ratings["rating"] - ratings.groupby("userId")["rating"].transform("mean"),
            idx=pos.loc[ratings["movieId"]].to_numpy(),
        )
        self._history = {
            u: (g["idx"].to_numpy(), g["peso"].to_numpy()) for u, g in r.groupby("userId")
        }
        return self

    def _score(self, user_id: int) -> pd.Series:
        if user_id not in self._history:
            return pd.Series(0.0, index=self._items)
        idx, pesos = self._history[user_id]
        if not np.any(pesos):  # usuário deu a mesma nota a tudo
            pesos = np.ones_like(pesos)
        perfil = pesos @ self._X[idx]  # vetor (1 x vocabulário)
        scores = self._X @ np.asarray(perfil).ravel()
        return pd.Series(scores, index=self._items)

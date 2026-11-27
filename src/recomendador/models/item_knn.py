"""Filtragem colaborativa item-item ("quem viu X também viu Y").

1. Monta a matriz esparsa usuário × filme.
2. Calcula a similaridade de cosseno entre as colunas (filmes): dois filmes são
   parecidos quando o mesmo grupo de pessoas os avaliou.
3. Para cada filme, guarda só os K vizinhos mais parecidos e com similaridade
   positiva (reduz ruído e memória).
4. A pontuação de um filme para o usuário é a soma das similaridades com os
   filmes que ele já avaliou.

implicito=True  → só importa SE a pessoa avaliou (1), não a nota.
implicito=False → usa a nota centrada na média do usuário (cosseno ajustado):
                  filmes que ele odiou "empurram" os vizinhos para baixo.
"""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

from .base import Recommender


class ItemKNNRecommender(Recommender):
    def __init__(self, k_vizinhos: int = 50, implicito: bool = True, bloco: int = 1000):
        self.k_vizinhos = k_vizinhos
        self.implicito = implicito
        self.bloco = bloco  # filmes processados por vez (controla o uso de memória)
        tipo = "implícito" if implicito else "notas"
        self.name = f"Item-item k={k_vizinhos} ({tipo})"

    def fit(self, ratings: pd.DataFrame) -> "ItemKNNRecommender":
        self._remember_seen(ratings)
        self._items = np.sort(ratings["movieId"].unique())
        users = np.sort(ratings["userId"].unique())
        self._user_pos = pd.Series(np.arange(len(users)), index=users)
        item_pos = pd.Series(np.arange(len(self._items)), index=self._items)
        self._item_pos = item_pos

        if self.implicito:
            valores = np.ones(len(ratings), dtype=np.float32)
        else:
            media = ratings.groupby("userId")["rating"].transform("mean")
            valores = (ratings["rating"] - media).to_numpy(dtype=np.float32)

        n_users, n_items = len(users), len(self._items)
        self._X = csr_matrix(
            (valores, (self._user_pos.loc[ratings["userId"]], item_pos.loc[ratings["movieId"]])),
            shape=(n_users, n_items),
        )
        self.sim_ = self._top_k_similarities(normalize(self._X, axis=0), n_items)
        return self

    def _top_k_similarities(self, Xn: csr_matrix, n_items: int) -> csr_matrix:
        """Similaridade item-item em blocos, mantendo só os K maiores por linha.

        A matriz completa (cerca de 9 mil × 9 mil) ocuparia centenas de MB;
        em blocos, o pico de memória fica em dezenas de MB.
        """
        k = min(self.k_vizinhos, n_items - 1)
        Xt = Xn.T.tocsr()
        linhas, colunas, dados = [], [], []
        for inicio in range(0, n_items, self.bloco):
            fim = min(inicio + self.bloco, n_items)
            sim = (Xt[inicio:fim] @ Xn).toarray()
            sim[np.arange(fim - inicio), np.arange(inicio, fim)] = 0.0  # ignora a si mesmo
            if k <= 0:
                continue
            top = np.argpartition(-sim, k - 1, axis=1)[:, :k]
            valores = np.take_along_axis(sim, top, axis=1)
            mask = valores > 0
            linhas.append(np.repeat(np.arange(inicio, fim), k).reshape(-1, k)[mask])
            colunas.append(top[mask])
            dados.append(valores[mask])
        if not dados:
            return csr_matrix((n_items, n_items), dtype=np.float32)
        return csr_matrix(
            (np.concatenate(dados), (np.concatenate(linhas), np.concatenate(colunas))),
            shape=(n_items, n_items),
        )

    def _score(self, user_id: int) -> pd.Series:
        if user_id not in self._user_pos.index:
            return pd.Series(0.0, index=self._items)
        historico = self._X[self._user_pos[user_id]]  # 1 × filmes
        scores = self.sim_ @ historico.T              # filmes × 1
        return pd.Series(scores.toarray().ravel(), index=self._items)

    def _history_vector(self, history: dict[int, float]) -> csr_matrix:
        filmes = [m for m in history if m in self._item_pos.index]
        valores = np.array([history[m] for m in filmes], dtype=np.float32)
        if self.implicito:
            valores = np.ones_like(valores)
        elif len(valores):
            centradas = valores - valores.mean()
            # notas todas iguais (ex.: só "curti") viram zero; nesse caso usa peso 1
            valores = centradas if np.any(centradas) else np.ones_like(valores)
        return csr_matrix(
            (valores, (np.zeros(len(filmes), dtype=int), self._item_pos.loc[filmes].to_numpy())),
            shape=(1, len(self._items)),
        )

    def _score_history(self, history: dict[int, float]) -> pd.Series:
        scores = self.sim_ @ self._history_vector(history).T
        return pd.Series(scores.toarray().ravel(), index=self._items)

    def explain(self, movie_id: int, history: dict[int, float]) -> int | None:
        """Qual filme do histórico mais "puxou" esta recomendação."""
        if movie_id not in self._item_pos.index:
            return None
        linha = self.sim_[self._item_pos[movie_id]].toarray().ravel()
        candidatos = {m: linha[self._item_pos[m]] for m in history if m in self._item_pos.index}
        melhor = max(candidatos, key=candidatos.get, default=None)
        return melhor if melhor is not None and candidatos[melhor] > 0 else None

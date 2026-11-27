import numpy as np
import pandas as pd

from recomendador.models import SVDRecommender


def _ratings():
    # Dois grupos de gosto: {1, 2} e {3, 4}.
    pares = [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2),
             (4, 3), (4, 4), (5, 3), (5, 4), (6, 1), (7, 3)]
    t = pd.Timestamp("2020-01-01")
    return pd.DataFrame([(u, m, 4.0, t) for u, m in pares],
                        columns=["userId", "movieId", "rating", "timestamp"])


def test_recomenda_filme_do_mesmo_grupo_de_gosto():
    model = SVDRecommender(fatores=2).fit(_ratings())
    assert model.recommend(user_id=6, k=1) == [2]
    assert model.recommend(user_id=7, k=1) == [4]


def test_formato_dos_fatores():
    model = SVDRecommender(fatores=2).fit(_ratings())
    assert model.V_.shape == (4, 2)  # 4 filmes × 2 fatores


def test_fatores_limitados_ao_tamanho_da_matriz():
    model = SVDRecommender(fatores=500).fit(_ratings())
    assert model.V_.shape[1] == 3  # no máximo min(usuários, filmes) - 1


def test_resultado_reproduzivel_com_mesma_semente():
    a = SVDRecommender(fatores=2, seed=1).fit(_ratings())._score(6)
    b = SVDRecommender(fatores=2, seed=1).fit(_ratings())._score(6)
    assert np.allclose(a, b)

import numpy as np
import pandas as pd

from recomendador.models import ItemKNNRecommender


def _ratings():
    # Quem viu o filme 1 quase sempre viu o 2; o 3 é visto por outro grupo.
    pares = [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2), (3, 4),
             (4, 3), (4, 4), (5, 3), (6, 1)]
    t = pd.Timestamp("2020-01-01")
    return pd.DataFrame(
        [(u, m, 4.0, t) for u, m in pares],
        columns=["userId", "movieId", "rating", "timestamp"],
    )


def test_quem_viu_1_recebe_2():
    model = ItemKNNRecommender(k_vizinhos=2).fit(_ratings())
    assert model.recommend(user_id=6, k=1) == [2]


def test_filme_nao_e_vizinho_de_si_mesmo():
    model = ItemKNNRecommender(k_vizinhos=3).fit(_ratings())
    assert np.all(model.sim_.diagonal() == 0)


def test_cada_filme_guarda_no_maximo_k_vizinhos():
    model = ItemKNNRecommender(k_vizinhos=1).fit(_ratings())
    assert model.sim_.getnnz(axis=1).max() <= 1


def test_blocos_pequenos_dao_o_mesmo_resultado():
    grande = ItemKNNRecommender(k_vizinhos=2, bloco=1000).fit(_ratings())
    pequeno = ItemKNNRecommender(k_vizinhos=2, bloco=1).fit(_ratings())
    assert (grande.sim_ != pequeno.sim_).nnz == 0


def test_modo_notas_penaliza_vizinhos_de_filme_odiado():
    # Usuários 1 e 2 tratam os filmes 3 e 5 do mesmo jeito: 3 e 5 são vizinhos.
    t = pd.Timestamp("2020-01-01")
    df = pd.DataFrame(
        [(1, 1, 5.0, t), (1, 3, 1.0, t), (1, 5, 1.0, t),
         (2, 1, 5.0, t), (2, 3, 1.0, t), (2, 5, 1.0, t),
         (3, 3, 1.0, t), (3, 4, 5.0, t)],
        columns=["userId", "movieId", "rating", "timestamp"],
    )
    notas = ItemKNNRecommender(k_vizinhos=3, implicito=False).fit(df)
    implicito = ItemKNNRecommender(k_vizinhos=3, implicito=True).fit(df)
    assert notas._score(3)[5] < 0      # odiou o 3, então o vizinho 5 é penalizado
    assert implicito._score(3)[5] > 0  # no modo implícito, ter visto já conta a favor

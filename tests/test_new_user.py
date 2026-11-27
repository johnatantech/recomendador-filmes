import pandas as pd
import pytest

from recomendador.models import (
    ContentRecommender, ItemKNNRecommender, PopularityRecommender, SVDRecommender,
)


def _ratings():
    # Dois grupos de gosto: {1, 2, 5} e {3, 4}.
    pares = [(1, 1), (1, 2), (1, 5), (2, 1), (2, 2), (2, 5), (3, 1), (3, 2),
             (4, 3), (4, 4), (5, 3), (5, 4), (6, 3), (6, 4)]
    t = pd.Timestamp("2020-01-01")
    return pd.DataFrame([(u, m, 4.0, t) for u, m in pares],
                        columns=["userId", "movieId", "rating", "timestamp"])


@pytest.mark.parametrize("model", [
    ItemKNNRecommender(k_vizinhos=3),
    SVDRecommender(fatores=2),
    SVDRecommender(fatores=2, implicito=False),
])
def test_usuario_novo_recebe_filmes_do_grupo_dele(model):
    model.fit(_ratings())
    recs = model.recommend_for_history({3: 5.0}, k=1)
    assert recs == [4]


def test_nao_recomenda_o_que_o_usuario_marcou():
    model = PopularityRecommender().fit(_ratings())
    assert 1 not in model.recommend_for_history({1: 5.0}, k=4)


def test_filme_fora_do_catalogo_e_ignorado():
    model = SVDRecommender(fatores=2).fit(_ratings())
    assert model.recommend_for_history({3: 5.0, 999: 5.0}, k=1) == [4]


def test_explicacao_aponta_o_filme_que_puxou_a_recomendacao():
    model = ItemKNNRecommender(k_vizinhos=3).fit(_ratings())
    assert model.explain(4, {3: 5.0, 1: 5.0}) == 3


def test_modelo_sem_suporte_avisa():
    movies = pd.DataFrame({"movieId": [1, 2, 3, 4, 5], "genres": ["Drama"] * 5})
    model = ContentRecommender(movies).fit(_ratings())
    with pytest.raises(NotImplementedError):
        model.recommend_for_history({1: 5.0})


def test_item_item_por_notas_com_usuario_novo():
    # Notas variadas: no modo por notas, nota constante não carrega informação.
    t = pd.Timestamp("2020-01-01")
    dados = [(1, 1, 5.0), (1, 2, 5.0), (1, 3, 1.0), (2, 1, 5.0), (2, 2, 4.5), (2, 4, 1.0),
             (3, 3, 5.0), (3, 4, 5.0), (3, 1, 1.0), (4, 3, 4.5), (4, 4, 5.0), (4, 2, 1.0)]
    df = pd.DataFrame([(u, m, r, t) for u, m, r in dados],
                      columns=["userId", "movieId", "rating", "timestamp"])
    model = ItemKNNRecommender(k_vizinhos=3, implicito=False).fit(df)
    assert model.recommend_for_history({3: 5.0, 1: 1.0}, k=1) == [4]

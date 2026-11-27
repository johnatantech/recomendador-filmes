import pandas as pd

from recomendador.evaluate import evaluate, evaluate_per_user
from recomendador.models import PopularityRecommender


def _dados():
    t = pd.Timestamp("2020-01-01")
    train = pd.DataFrame({"userId": [1, 2, 3, 3], "movieId": [10, 10, 20, 10],
                          "rating": [5.0] * 4, "timestamp": [t] * 4})
    test = pd.DataFrame({"userId": [1, 2], "movieId": [20, 30], "rating": [5.0, 5.0],
                         "timestamp": [t] * 2})
    return train, test


def test_uma_linha_por_usuario_com_relevantes():
    train, test = _dados()
    per_user = evaluate_per_user(PopularityRecommender().fit(train), test, k=1)
    assert list(per_user.index) == [1, 2]
    assert per_user.loc[1, "precision"] == 1.0  # recebeu o 20, que estava no teste
    assert per_user.loc[2, "precision"] == 0.0


def test_resumo_e_a_media_dos_usuarios():
    train, test = _dados()
    assert evaluate(PopularityRecommender().fit(train), train, test, k=1)["precision@1"] == 0.5

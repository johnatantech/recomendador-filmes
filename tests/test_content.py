import pandas as pd

from recomendador.models import ContentRecommender
from recomendador.split import remove_test_tags

MOVIES = pd.DataFrame({
    "movieId": [1, 2, 3, 4, 5],
    "genres": ["Animation|Children", "Animation|Comedy", "Horror|Thriller",
               "Horror", "Animation|Children|Comedy"],
})


def _ratings():
    t = pd.Timestamp("2020-01-01")
    return pd.DataFrame({
        "userId":  [1, 1, 1, 2, 2, 2],
        "movieId": [1, 2, 3, 3, 4, 5],
        "rating":  [5.0, 5.0, 1.0, 4.0, 4.0, 4.0],
        "timestamp": [t] * 6,
    })


def test_quem_gosta_de_animacao_recebe_animacao():
    model = ContentRecommender(MOVIES).fit(_ratings())
    recs = model.recommend(user_id=1, k=2)
    assert recs[0] == 5          # animação infantil, parecido com o que ele amou
    assert 4 not in recs[:1]     # terror, parecido com o que ele odiou


def test_notas_iguais_nao_quebram_o_perfil():
    model = ContentRecommender(MOVIES).fit(_ratings())
    assert len(model.recommend(user_id=2, k=3)) == 2  # só 1 e 2 não foram vistos


def test_tags_entram_no_vocabulario():
    tags = pd.DataFrame({"userId": [9], "movieId": [1], "tag": ["Pixar"],
                         "timestamp": [pd.Timestamp("2020-01-01")]})
    model = ContentRecommender(MOVIES, tags).fit(_ratings())
    assert "pixar" in model.vectorizer_.vocabulary_


def test_remove_tags_de_filmes_do_teste_do_proprio_usuario():
    tags = pd.DataFrame({"userId": [1, 2], "movieId": [5, 5], "tag": ["a", "b"]})
    test = pd.DataFrame({"userId": [1], "movieId": [5], "rating": [4.0]})
    restantes = remove_test_tags(tags, test)
    assert list(restantes["userId"]) == [2]  # tag de outro usuário continua

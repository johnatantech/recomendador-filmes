"""Uso: python scripts/comparar.py"""
import pandas as pd

from recomendador.data import load, load_tags
from recomendador.evaluate import evaluate
from recomendador.models import ContentRecommender, PopularityRecommender, RandomRecommender
from recomendador.split import remove_test_tags, temporal_leave_last_out


def main() -> None:
    ratings, movies = load()
    train, test = temporal_leave_last_out(ratings, n_test=5)
    tags = remove_test_tags(load_tags(), test)

    modelos = [
        RandomRecommender(),
        PopularityRecommender("contagem"),
        PopularityRecommender("media_ponderada"),
        ContentRecommender(movies),
        ContentRecommender(movies, tags),
    ]
    resultados = [evaluate(m.fit(train), train, test, k=10) for m in modelos]
    print(pd.DataFrame(resultados).to_string(index=False))


if __name__ == "__main__":
    main()

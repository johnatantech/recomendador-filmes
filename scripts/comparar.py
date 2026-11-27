"""Uso: python scripts/comparar.py"""
import time

import pandas as pd

from recomendador.data import load, load_tags
from recomendador.evaluate import evaluate
from recomendador.models import (
    ContentRecommender, ItemKNNRecommender, PopularityRecommender, RandomRecommender,
)
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
        ItemKNNRecommender(k_vizinhos=20),
        ItemKNNRecommender(k_vizinhos=50),
        ItemKNNRecommender(k_vizinhos=200),
        ItemKNNRecommender(k_vizinhos=50, implicito=False),
    ]
    resultados = []
    for m in modelos:
        inicio = time.perf_counter()
        resultado = evaluate(m.fit(train), train, test, k=10)
        resultado["tempo_s"] = round(time.perf_counter() - inicio, 1)
        resultados.append(resultado)
        print(f"✓ {m.name}")
    print()
    print(pd.DataFrame(resultados).to_string(index=False))


if __name__ == "__main__":
    main()

"""Uso: python scripts/comparar.py"""
import pandas as pd

from recomendador.data import load
from recomendador.evaluate import evaluate
from recomendador.models import PopularityRecommender, RandomRecommender
from recomendador.split import temporal_leave_last_out


def main() -> None:
    ratings, _ = load()
    train, test = temporal_leave_last_out(ratings, n_test=5)
    modelos = [
        RandomRecommender(),
        PopularityRecommender("contagem"),
        PopularityRecommender("media_ponderada"),
    ]
    resultados = [evaluate(m.fit(train), train, test, k=10) for m in modelos]
    print(pd.DataFrame(resultados).to_string(index=False))


if __name__ == "__main__":
    main()

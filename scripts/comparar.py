"""Uso: python scripts/comparar.py"""
import time

import pandas as pd

from recomendador.data import load, load_tags
from recomendador.evaluate import evaluate_per_user, summarize
from recomendador.models import (
    ContentRecommender, ItemKNNRecommender, PopularityRecommender,
    RandomRecommender, SVDRecommender,
)
from recomendador.split import remove_test_tags, temporal_leave_last_out
from recomendador.stats import bootstrap_ci, paired_bootstrap

K = 10
METRICA = "ndcg"  # métrica principal das comparações
ALFA = 0.05
REFERENCIA = "Popularidade (contagem)"
PARES_EXTRAS = [  # perguntas específicas além de "vence a referência?"
    ("Item-item k=200 (implícito)", "Item-item k=50 (implícito)"),
    ("Item-item k=200 (implícito)", "Item-item k=20 (implícito)"),
    ("Item-item k=50 (implícito)", "Item-item k=50 (notas)"),
    ("Conteúdo (gêneros + tags)", "Conteúdo (gêneros)"),
    ("SVD 50 fatores (implícito)", "Item-item k=200 (implícito)"),
    ("SVD 100 fatores (implícito)", "SVD 20 fatores (implícito)"),
    ("SVD 50 fatores (implícito)", "SVD 50 fatores (notas)"),
    ("SVD 50 fatores (notas)", "Item-item k=200 (implícito)"),
]


def main() -> None:
    ratings, movies = load()
    train, test = temporal_leave_last_out(ratings, n_test=5)
    tags = remove_test_tags(load_tags(), test)
    n_items = train["movieId"].nunique()

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
        SVDRecommender(fatores=20),
        SVDRecommender(fatores=50),
        SVDRecommender(fatores=100),
        SVDRecommender(fatores=50, implicito=False),
    ]

    por_usuario, resumos = {}, []
    for m in modelos:
        inicio = time.perf_counter()
        per_user = evaluate_per_user(m.fit(train), test, K)
        resumo = summarize(per_user, m.name, n_items, K)
        _, inf, sup = bootstrap_ci(per_user[METRICA])
        resumo[f"IC95% {METRICA}"] = f"[{inf:.4f}, {sup:.4f}]"
        resumo["tempo_s"] = round(time.perf_counter() - inicio, 1)
        por_usuario[m.name] = per_user
        resumos.append(resumo)
        print(f"✓ {m.name}")

    print("\n=== Resultados ===")
    print(pd.DataFrame(resumos).to_string(index=False))

    pares = [(nome, REFERENCIA) for nome in por_usuario if nome != REFERENCIA] + PARES_EXTRAS
    alfa_bonferroni = ALFA / len(pares)
    comparacoes = []
    for a, b in pares:
        r = paired_bootstrap(por_usuario[a][METRICA], por_usuario[b][METRICA])
        comparacoes.append({
            "modelo A": a,
            "modelo B": b,
            f"Δ {METRICA}@{K}": round(r["diferenca"], 4),
            "IC95%": f"[{r['ic_inf']:+.4f}, {r['ic_sup']:+.4f}]",
            "p": round(r["p_valor"], 3),
            "resultado": ("A melhor" if r["diferenca"] > 0 else "B melhor")
            if r["significativo"] else "empate (inconclusivo)",
            "robusto": "sim" if r["significativo"] and r["p_valor"] < alfa_bonferroni else "não",
        })

    print(f"\n=== Comparações pareadas ({METRICA}@{K}, bootstrap por usuário) ===")
    print(pd.DataFrame(comparacoes).to_string(index=False))
    print(f"\nrobusto = significativo mesmo com correção de Bonferroni "
          f"({len(pares)} comparações → p < {alfa_bonferroni:.4f})")


if __name__ == "__main__":
    main()

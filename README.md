# 🎬 Recomendador de Filmes: comparando abordagens

Compara diferentes estratégias de recomendação sobre a base pública
[MovieLens](https://grouplens.org/datasets/movielens/) (`ml-latest-small`,
cerca de 100 mil avaliações), usando **a mesma divisão de dados e as mesmas métricas**
para todas, de forma justa e reproduzível.

## Pergunta central
> Quanto uma abordagem mais sofisticada realmente ganha sobre um baseline simples?

## Abordagens
| # | Modelo | Ideia | Status |
|---|--------|-------|--------|
| 0 | Aleatório | Piso de referência | ✅ |
| 1 | Popularidade (contagem / média bayesiana) | O que todo mundo vê | ✅ |
| 2 | Conteúdo (TF-IDF de gêneros e tags) | Filmes parecidos com os que você gostou | ⏳ |
| 3 | Filtragem colaborativa item-item | Quem gostou de X também gostou de Y | ⏳ |
| 4 | Fatoração de matrizes (SVD) | Gostos latentes de usuários e filmes | ⏳ |

## Metodologia
- **Divisão temporal por usuário**: as últimas 5 avaliações de cada usuário vão para teste,
  simulando "o que a pessoa vai assistir depois".
- **Relevante** = nota ≥ 4 no teste.
- **Métricas**: Precision@10, Recall@10, NDCG@10 (posição importa) e **cobertura**
  (quanto do catálogo o modelo consegue recomendar).

## Como rodar
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
python scripts/comparar.py   # baixa a base na primeira execução
```

## Estrutura
```
src/recomendador/
├── data.py          # download e carga da MovieLens
├── split.py         # divisão temporal treino/teste
├── metrics.py       # métricas de ranking
├── evaluate.py      # avaliação padronizada
└── models/
    ├── base.py      # interface comum (fit / recommend)
    └── popularity.py
scripts/comparar.py  # tabela comparativa
tests/
```

## Decisões técnicas
- **Interface única (`Recommender`)**: cada modelo implementa `fit()` e `_score()`;
  a avaliação não muda quando entra um modelo novo.
- **Baselines primeiro**: um modelo complexo só se justifica se vencer o aleatório e a popularidade.
- **Base fora do Git**: a licença do GroupLens não permite redistribuição; o download é automático.

## Licença dos dados
F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets: History and Context.*
ACM TiiS 5, 4. Uso conforme os termos do GroupLens.

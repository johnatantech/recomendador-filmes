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
| 2 | Conteúdo (TF-IDF de gêneros e tags) | Filmes parecidos com os que você gostou | ✅ |
| 3 | Filtragem colaborativa item-item | Quem gostou de X também gostou de Y | ⏳ |
| 4 | Fatoração de matrizes (SVD) | Gostos latentes de usuários e filmes | ⏳ |

## Metodologia
- **Divisão temporal por usuário**: as últimas 5 avaliações de cada usuário vão para teste,
  simulando "o que a pessoa vai assistir depois".
- **Relevante** = nota ≥ 4 no teste.
- **Métricas**: Precision@10, Recall@10, NDCG@10 (posição importa) e **cobertura**
  (quanto do catálogo o modelo consegue recomendar).
- **Sem vazamento de tags**: tags que um usuário aplicou aos filmes do próprio teste são
  removidas antes do treino, para o modelo de conteúdo não "espiar" a resposta.

## Como rodar
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
python scripts/comparar.py   # baixa a base na primeira execução
```

## Resultados

### Fase 1: baselines

| Modelo | Precision@10 | Recall@10 | NDCG@10 | Cobertura |
|--------|-------------:|----------:|--------:|----------:|
| Aleatório | 0,0000 | 0,0000 | 0,0000 | 43,0% |
| Popularidade (contagem) | 0,0173 | 0,0492 | 0,0396 | 1,2% |
| Popularidade (média bayesiana) | 0,0170 | 0,0503 | 0,0378 | 0,8% |

*542 usuários avaliados, K = 10.*

**O que os números dizem**
- **O aleatório zera.** Com cerca de 3 filmes relevantes por usuário em um catálogo de
  aproximadamente 9 mil, acertar por sorte é praticamente impossível. Isso confirma que a
  avaliação não premia o acaso.
- **A popularidade vence o acaso, mas com números absolutos baixos.** Há dois motivos: o teto
  de precision fica perto de 0,3 (poucos relevantes por usuário) e a divisão temporal é
  exigente, porque as últimas avaliações tendem a ser de filmes recentes ou de nicho, ainda
  pouco populares no treino. Um split aleatório daria números maiores, porém irreais.
- **Os dois critérios de popularidade empatam.** Com o quantil de 0,9, a média bayesiana
  exige tantos votos que acaba favorecendo os mesmos filmes muito avaliados.
- **A cobertura é o achado central.** A popularidade recomenda cerca de 1% do catálogo:
  todos recebem quase a mesma lista. Esse é o dilema **precisão × diversidade** que os
  próximos modelos precisam enfrentar.

### Fase 2: conteúdo (TF-IDF)

**Hipótese registrada antes de rodar:** a cobertura deve subir muito, porque cada usuário
recebe uma lista personalizada pelos gêneros e tags que prefere. A precisão pode cair, já que
"parecido com o que você gostou" não é o mesmo que "o que você vai assistir em seguida".

*Resultados: a preencher após `python scripts/comparar.py`.*

## Estrutura
```
src/recomendador/
├── data.py          # download e carga da MovieLens
├── split.py         # divisão temporal treino/teste
├── metrics.py       # métricas de ranking
├── evaluate.py      # avaliação padronizada
└── models/
    ├── base.py      # interface comum (fit / recommend)
    ├── popularity.py
    └── content.py   # TF-IDF de gêneros e tags
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

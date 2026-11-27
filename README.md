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
| 3 | Filtragem colaborativa item-item | Quem gostou de X também gostou de Y | ✅ |
| 4 | Fatoração de matrizes (SVD) | Gostos latentes de usuários e filmes | ✅ |

## Metodologia
- **Divisão temporal por usuário**: as últimas 5 avaliações de cada usuário vão para teste,
  simulando "o que a pessoa vai assistir depois".
- **Relevante** = nota ≥ 4 no teste.
- **Métricas**: Precision@10, Recall@10, NDCG@10 (posição importa) e **cobertura**
  (quanto do catálogo o modelo consegue recomendar).
- **Intervalos de confiança (bootstrap por usuário)**: cada métrica vem com um IC de 95%, e os
  modelos são comparados de forma **pareada** (nos mesmos usuários). Uma diferença só é
  considerada real quando o intervalo da diferença não contém zero.
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

| Modelo | Precision@10 | Recall@10 | NDCG@10 | Cobertura |
|--------|-------------:|----------:|--------:|----------:|
| Conteúdo (gêneros) | 0,0013 | 0,0035 | 0,0026 | 19,3% |
| Conteúdo (gêneros + tags) | 0,0022 | 0,0074 | 0,0045 | 20,1% |

**Hipótese confirmada.** A cobertura subiu cerca de 16 vezes em relação à popularidade, mas a
precisão caiu cerca de 8 vezes. Os gêneros são genéricos demais (milhares de filmes empatam)
e o modelo ignora o viés de popularidade: recomenda títulos obscuros que combinam "no papel",
mas que pouca gente assiste. As tags melhoram o resultado, porém a diferença (cerca de 7 para
12 acertos em mais de 5 mil recomendações) é pequena demais para conclusões firmes.

### Fase 3: filtragem colaborativa item-item

**Hipótese registrada antes de rodar:** o item-item deve superar a popularidade em precisão,
porque usa comportamento real ("quem viu X também viu Y") em vez de rótulos, mantendo uma
cobertura intermediária entre popularidade e conteúdo. Esperamos que:
- **K pequeno** (poucos vizinhos) aumente a cobertura e o ruído; **K grande** aproxime o
  modelo da popularidade;
- o **modo implícito** (só "avaliou ou não") vença o **modo por notas**, já que a tarefa é
  prever *o que* a pessoa vai assistir, não *quanto* vai gostar.

| Modelo | Precision@10 | Recall@10 | NDCG@10 | Cobertura | Tempo |
|--------|-------------:|----------:|--------:|----------:|------:|
| Item-item k=20 (implícito) | 0,0256 | 0,0767 | 0,0514 | 8,0% | 2,1 s |
| Item-item k=50 (implícito) | 0,0247 | 0,0780 | 0,0480 | 7,6% | 2,2 s |
| Item-item k=200 (implícito) | 0,0264 | 0,0824 | 0,0546 | 7,0% | 3,9 s |
| Item-item k=50 (notas) | 0,0220 | 0,0655 | 0,0506 | 12,3% | 2,3 s |

**Primeiro modelo a empatar com a popularidade em acerto, com 6 vezes mais diversidade.**
Nas médias, o melhor item-item (k=200) fica cerca de 53% acima em precision, 67% em recall e
38% em NDCG. Mas o teste de significância (abaixo) mostra que essa vantagem tem evidência
fraca; o ganho de cobertura, por outro lado, é claro.

- **K e cobertura:** confirmado. Quanto mais vizinhos, mais o modelo converge para os filmes
  populares (cobertura de 8,0% para 7,0%).
- **K e precisão:** inconclusivo. As diferenças equivalem a poucos acertos e podem ser ruído.
- **Implícito × notas:** parcialmente confirmado. O implícito vence em precision e recall,
  mas o modo por notas empata no NDCG e tem a maior cobertura entre os colaborativos (12,3%):
  as notas trocam um pouco de acerto por diversidade.

### Significância estatística

Comparações pareadas do NDCG@10 por bootstrap (2.000 reamostragens de usuários), fases 1 a 3:

| Comparação | Δ NDCG@10 | IC 95% | p | Resultado |
|------------|----------:|-------:|--:|-----------|
| Aleatório × Popularidade | −0,0396 | [−0,0495; −0,0308] | <0,001 | Popularidade melhor |
| Pop. bayesiana × Pop. contagem | −0,0018 | [−0,0104; +0,0065] | 0,658 | Empate |
| Conteúdo (gêneros) × Popularidade | −0,0370 | [−0,0474; −0,0278] | <0,001 | Popularidade melhor |
| Conteúdo (+ tags) × Popularidade | −0,0351 | [−0,0457; −0,0260] | <0,001 | Popularidade melhor |
| Item-item k=20 × Popularidade | +0,0118 | [−0,0004; +0,0237] | 0,055 | Empate |
| Item-item k=50 × Popularidade | +0,0085 | [−0,0033; +0,0198] | 0,165 | Empate |
| Item-item k=200 × Popularidade | +0,0150 | [+0,0014; +0,0278] | 0,029 | Item-item melhor* |
| Item-item notas × Popularidade | +0,0110 | [−0,0012; +0,0224] | 0,081 | Empate |
| Item-item k=200 × k=50 | +0,0066 | [−0,0009; +0,0141] | 0,089 | Empate |
| Item-item k=200 × k=20 | +0,0032 | [−0,0065; +0,0126] | 0,515 | Empate |
| Item-item implícito × notas | −0,0025 | [−0,0145; +0,0096] | 0,720 | Empate |
| Conteúdo + tags × só gêneros | +0,0019 | [−0,0017; +0,0051] | 0,267 | Empate |

\* *Não sobrevive à correção de Bonferroni para 12 comparações (p < 0,004).*

**O que é sólido:** o aleatório e os modelos de conteúdo são piores que a popularidade, com
folga. **O que não é:** a vantagem do item-item em acerto tem evidência fraca, e as diferenças
entre valores de K, entre modos e entre variantes de conteúdo são indistinguíveis de ruído.

**Por que é tão difícil separar os modelos:** são só 542 usuários, com cerca de 3 filmes
relevantes cada, e acertos são raros. Isso gera muita variância. Uma base maior (como a
MovieLens 1M, com 6 mil usuários) daria poder estatístico para conclusões mais firmes.

> **Cuidado com comparações múltiplas:** quanto mais comparações a 95% de confiança, maior a
> chance de alguma parecer significativa por acaso. O `comparar.py` marca como **robustas**
> só as diferenças que sobrevivem à correção de Bonferroni.

### Fase 4: fatoração de matrizes (SVD)

**Hipótese registrada antes de rodar:** o SVD deve ficar próximo do item-item em acerto, já que
os dois exploram o mesmo sinal (coocorrência de filmes), mas com cobertura menor: poucos fatores
capturam principalmente os padrões dominantes, que são os dos filmes populares. Esperamos que
**mais fatores** aumentem a cobertura e que, dado o poder estatístico limitado, a maioria das
diferenças contra o item-item termine em empate.

*Resultados: a preencher após `python scripts/comparar.py`.*

## Estrutura
```
src/recomendador/
├── data.py          # download e carga da MovieLens
├── split.py         # divisão temporal treino/teste
├── metrics.py       # métricas de ranking
├── evaluate.py      # avaliação padronizada (agregada e por usuário)
├── stats.py         # bootstrap e comparações pareadas
└── models/
    ├── base.py      # interface comum (fit / recommend)
    ├── popularity.py
    ├── content.py   # TF-IDF de gêneros e tags
    ├── item_knn.py  # colaborativa item-item com K vizinhos
    └── svd.py       # fatoração de matrizes (PureSVD)
scripts/comparar.py  # tabela comparativa
tests/
```

## Decisões técnicas
- **Interface única (`Recommender`)**: cada modelo implementa `fit()` e `_score()`;
  a avaliação não muda quando entra um modelo novo.
- **Baselines primeiro**: um modelo complexo só se justifica se vencer o aleatório e a popularidade.
- **Similaridade em blocos**: a matriz item-item completa ocuparia centenas de MB; calculada
  em blocos e mantendo só os K vizinhos de cada filme, o pico de memória fica bem menor.
- **Base fora do Git**: a licença do GroupLens não permite redistribuição; o download é automático.

## Licença dos dados
F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets: History and Context.*
ACM TiiS 5, 4. Uso conforme os termos do GroupLens.

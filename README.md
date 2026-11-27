# 🎬 Recomendador de Filmes: comparando abordagens

Compara quatro famílias de algoritmos de recomendação sobre a base pública
[MovieLens](https://grouplens.org/datasets/movielens/) (`ml-latest-small`, cerca de 100 mil
avaliações de 610 usuários), usando **a mesma divisão de dados, as mesmas métricas e testes de
significância estatística** para todas.

[![Abrir no Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://recomendador-filmes-jf.streamlit.app)

🔗 **Demo:** [recomendador-filmes-jf.streamlit.app](https://recomendador-filmes-jf.streamlit.app)

## Pergunta central
> Quanto uma abordagem mais sofisticada realmente ganha sobre um baseline simples?

## Resumo dos resultados

| Modelo (melhor variante) | NDCG@10 | Precision@10 | Cobertura | Vence a popularidade? |
|--------------------------|--------:|-------------:|----------:|-----------------------|
| Popularidade | 0,0396 | 0,0173 | 1,2% | *(referência)* |
| Conteúdo (gêneros + tags) | 0,0045 | 0,0022 | 20,1% | ❌ Perde (robusto) |
| Item-item k=200 | 0,0546 | 0,0264 | 7,0% | ⚠️ Evidência fraca |
| **SVD 50 fatores (notas)** | **0,0685** | **0,0327** | 6,8% | ✅ **Vence (robusto)** |

**O SVD por notas é o único modelo que supera a popularidade com robustez estatística:** +73%
de NDCG, quase o dobro de precision e recall, recomendando um catálogo cerca de 6 vezes mais
variado. "Robusto" significa significativo mesmo após a correção de Bonferroni para todas as
comparações feitas.

## Abordagens
| # | Modelo | Ideia | Status |
|---|--------|-------|--------|
| 0 | Aleatório | Piso de referência | ✅ |
| 1 | Popularidade (contagem / média bayesiana) | O que todo mundo vê | ✅ |
| 2 | Conteúdo (TF-IDF de gêneros e tags) | Filmes parecidos com os que você gostou | ✅ |
| 3 | Filtragem colaborativa item-item | Quem gostou de X também gostou de Y | ✅ |
| 4 | Fatoração de matrizes (PureSVD) | Gostos latentes de usuários e filmes | ✅ |
| 5 | App interativo (Streamlit) | Compare os modelos com os seus filmes | ✅ |

## Metodologia
- **Divisão temporal por usuário**: as últimas 5 avaliações de cada usuário vão para teste,
  simulando "o que a pessoa vai assistir depois". Um split aleatório daria números maiores,
  porém irreais.
- **Relevante** = nota ≥ 4 no teste.
- **Métricas**: Precision@10, Recall@10, NDCG@10 (posição importa) e **cobertura**
  (quanto do catálogo o modelo consegue recomendar).
- **Hipóteses registradas antes de cada execução**, para evitar explicar os resultados depois
  de vê-los.
- **Intervalos de confiança (bootstrap por usuário)**: cada métrica vem com um IC de 95%, e os
  modelos são comparados de forma **pareada** (nos mesmos usuários). Uma diferença só é
  considerada real quando o intervalo da diferença não contém zero.
- **Correção para comparações múltiplas**: com dezenas de testes, alguns parecem significativos
  por acaso. Só são chamadas de **robustas** as diferenças que sobrevivem à correção de
  Bonferroni.
- **Sem vazamento de tags**: tags que um usuário aplicou aos filmes do próprio teste são
  removidas antes do treino, para o modelo de conteúdo não "espiar" a resposta.

## Resultados por fase

### Fase 1: baselines

| Modelo | Precision@10 | Recall@10 | NDCG@10 | Cobertura |
|--------|-------------:|----------:|--------:|----------:|
| Aleatório | 0,0000 | 0,0000 | 0,0000 | 43,0% |
| Popularidade (contagem) | 0,0173 | 0,0492 | 0,0396 | 1,2% |
| Popularidade (média bayesiana) | 0,0170 | 0,0503 | 0,0378 | 0,8% |

- **O aleatório zera.** Com cerca de 3 filmes relevantes por usuário em um catálogo de
  aproximadamente 9 mil, acertar por sorte é praticamente impossível. Isso confirma que a
  avaliação não premia o acaso.
- **A popularidade vence o acaso, mas com números absolutos baixos.** O teto de precision fica
  perto de 0,3 (poucos relevantes por usuário) e a divisão temporal é exigente: as últimas
  avaliações tendem a ser de filmes recentes ou de nicho, ainda pouco populares no treino.
- **Os dois critérios de popularidade empatam.** Com o quantil de 0,9, a média bayesiana exige
  tantos votos que acaba favorecendo os mesmos filmes muito avaliados.
- **A cobertura é o achado central.** A popularidade recomenda cerca de 1% do catálogo: todos
  recebem quase a mesma lista. Esse é o dilema **precisão × diversidade** que os próximos
  modelos precisam enfrentar.

### Fase 2: conteúdo (TF-IDF)

**Hipótese:** a cobertura deve subir muito, porque cada usuário recebe uma lista personalizada
pelos gêneros e tags que prefere. A precisão pode cair, já que "parecido com o que você gostou"
não é o mesmo que "o que você vai assistir em seguida".

| Modelo | Precision@10 | Recall@10 | NDCG@10 | Cobertura |
|--------|-------------:|----------:|--------:|----------:|
| Conteúdo (gêneros) | 0,0013 | 0,0035 | 0,0026 | 19,3% |
| Conteúdo (gêneros + tags) | 0,0022 | 0,0074 | 0,0045 | 20,1% |

**Hipótese confirmada.** A cobertura subiu cerca de 16 vezes, mas a precisão caiu cerca de 8
vezes. Os gêneros são genéricos demais (milhares de filmes empatam) e o modelo ignora o viés de
popularidade: recomenda títulos obscuros que combinam "no papel", mas que pouca gente assiste.
O ganho das tags não é estatisticamente significativo.

### Fase 3: filtragem colaborativa item-item

**Hipótese:** o item-item deve superar a popularidade em precisão, porque usa comportamento real
em vez de rótulos, com cobertura intermediária. K pequeno deve aumentar a cobertura; K grande,
aproximar o modelo da popularidade; o modo implícito deve vencer o modo por notas.

| Modelo | Precision@10 | Recall@10 | NDCG@10 | Cobertura | Tempo |
|--------|-------------:|----------:|--------:|----------:|------:|
| Item-item k=20 (implícito) | 0,0256 | 0,0767 | 0,0514 | 8,0% | 2,1 s |
| Item-item k=50 (implícito) | 0,0247 | 0,0780 | 0,0480 | 7,6% | 2,2 s |
| Item-item k=200 (implícito) | 0,0264 | 0,0824 | 0,0546 | 7,0% | 3,9 s |
| Item-item k=50 (notas) | 0,0220 | 0,0655 | 0,0506 | 12,3% | 2,3 s |

**Empata com a popularidade em acerto, com cerca de 6 vezes mais diversidade.** Nas médias, o
k=200 fica 38% acima em NDCG, mas a evidência estatística é fraca (p = 0,029, sem sobreviver à
correção). A cobertura cai conforme K sobe (confirmado); as diferenças de acerto entre valores
de K e entre os modos são indistinguíveis de ruído.

### Fase 4: fatoração de matrizes (SVD)

**Hipótese:** o SVD deve ficar próximo do item-item em acerto, já que os dois exploram o mesmo
sinal (coocorrência), com cobertura menor; mais fatores devem aumentar a cobertura.

| Modelo | Precision@10 | Recall@10 | NDCG@10 | Cobertura | Tempo |
|--------|-------------:|----------:|--------:|----------:|------:|
| SVD 20 fatores (implícito) | 0,0266 | 0,0805 | 0,0596 | 5,4% | 0,9 s |
| SVD 50 fatores (implícito) | 0,0271 | 0,0840 | 0,0570 | 7,2% | 1,8 s |
| SVD 100 fatores (implícito) | 0,0280 | 0,0906 | 0,0590 | 8,5% | 3,1 s |
| **SVD 50 fatores (notas)** | **0,0327** | **0,1007** | **0,0685** | 6,8% | 1,8 s |

- **SVD × item-item:** confirmado o empate entre as versões implícitas (p = 0,614). Já o SVD
  por notas supera o melhor item-item (+0,0139 de NDCG, p = 0,008), mas sem sobreviver à
  correção de Bonferroni: há indício de vantagem, não prova.
- **Fatores e cobertura:** confirmado (5,4% → 8,5%), sem diferença significativa de acerto
  entre 20 e 100 fatores. A cobertura só fica abaixo da do item-item com poucos fatores.
- **Surpresa: notas × implícito.** No SVD, as notas vencem com robustez, ao contrário do
  item-item, onde os modos empataram. Sem centralização na média, a nota atua como peso de
  intensidade (nota 5 pesa mais que nota 2), sem os valores negativos que geram ruído no
  cosseno ajustado. É a configuração original do PureSVD (Cremonesi, Koren e Turrin, 2010).

### Significância estatística

Comparações pareadas do NDCG@10 por bootstrap (2.000 reamostragens de usuários). Correção de
Bonferroni para 20 comparações: **p < 0,0025**.

| Comparação | Δ NDCG@10 | IC 95% | p | Resultado |
|------------|----------:|-------:|--:|-----------|
| Aleatório × Popularidade | −0,0396 | [−0,0495; −0,0308] | <0,001 | Popularidade melhor (robusto) |
| Pop. bayesiana × Pop. contagem | −0,0018 | [−0,0104; +0,0065] | 0,658 | Empate |
| Conteúdo (gêneros) × Popularidade | −0,0370 | [−0,0474; −0,0278] | <0,001 | Popularidade melhor (robusto) |
| Conteúdo (+ tags) × Popularidade | −0,0351 | [−0,0457; −0,0260] | <0,001 | Popularidade melhor (robusto) |
| Item-item k=20 × Popularidade | +0,0118 | [−0,0004; +0,0237] | 0,055 | Empate |
| Item-item k=50 × Popularidade | +0,0085 | [−0,0033; +0,0198] | 0,165 | Empate |
| Item-item k=200 × Popularidade | +0,0150 | [+0,0014; +0,0278] | 0,029 | Item-item melhor (não robusto) |
| Item-item notas × Popularidade | +0,0110 | [−0,0012; +0,0224] | 0,081 | Empate |
| SVD 20 × Popularidade | +0,0200 | [+0,0085; +0,0312] | <0,001 | SVD melhor (robusto) |
| SVD 50 × Popularidade | +0,0174 | [+0,0054; +0,0289] | 0,005 | SVD melhor (não robusto) |
| SVD 100 × Popularidade | +0,0194 | [+0,0067; +0,0316] | 0,001 | SVD melhor (robusto) |
| **SVD 50 notas × Popularidade** | **+0,0289** | **[+0,0164; +0,0416]** | **<0,001** | **SVD melhor (robusto)** |
| Item-item k=200 × k=50 | +0,0066 | [−0,0009; +0,0141] | 0,089 | Empate |
| Item-item k=200 × k=20 | +0,0032 | [−0,0065; +0,0126] | 0,515 | Empate |
| Item-item implícito × notas | −0,0025 | [−0,0145; +0,0096] | 0,720 | Empate |
| Conteúdo + tags × só gêneros | +0,0019 | [−0,0017; +0,0051] | 0,267 | Empate |
| SVD 50 × Item-item k=200 | +0,0023 | [−0,0068; +0,0115] | 0,614 | Empate |
| SVD 100 × SVD 20 | −0,0006 | [−0,0097; +0,0089] | 0,932 | Empate |
| SVD 50 implícito × notas | −0,0115 | [−0,0179; −0,0054] | <0,001 | Notas melhor (robusto) |
| SVD 50 notas × Item-item k=200 | +0,0139 | [+0,0039; +0,0249] | 0,008 | SVD melhor (não robusto) |

**Por que é tão difícil separar os modelos:** são só 542 usuários, com cerca de 3 filmes
relevantes cada, e acertos são raros. Isso gera muita variância e explica a quantidade de
empates entre variantes próximas.

## Conclusões

1. **Baselines importam.** O modelo de conteúdo, apesar de intuitivo, perdeu feio para a simples
   popularidade. Sem um baseline, ele pareceria razoável.
2. **Comportamento vence rótulos.** Modelos que usam coocorrência (item-item e SVD) superam os
   que usam gêneros e tags.
3. **Precisão × diversidade é um trade-off real.** A popularidade acerta, mas recomenda 1% do
   catálogo; o conteúdo diversifica, mas erra; o SVD encontra o melhor equilíbrio.
4. **Médias enganam.** O item-item parecia 38% melhor que a popularidade, mas o bootstrap mostrou
   evidência fraca. Só as variantes do SVD resistiram aos testes mais rigorosos. Entre os dois
   melhores modelos, o SVD por notas tem indício de vantagem sobre o item-item (p = 0,008), mas
   sem robustez.
5. **Limitação:** com 542 usuários, o poder estatístico é baixo. Uma base maior permitiria
   distinguir variantes que aqui empataram.

## App interativo

**Acesse:** [recomendador-filmes-jf.streamlit.app](https://recomendador-filmes-jf.streamlit.app)
(no plano gratuito, o app pode estar "dormindo"; ele volta em cerca de um minuto).

O app permite marcar filmes de que você gosta e comparar, lado a lado, as recomendações da
popularidade, do item-item (com a explicação "porque você marcou X") e do SVD. Ele também mostra
quantos filmes se repetem entre as listas, o que deixa visível na prática o trade-off entre
precisão e diversidade.

Os modelos são treinados uma única vez com a base completa e recomendam para o usuário novo
sem retreinar (*fold-in*): o item-item soma as similaridades com os filmes marcados, e o SVD
projeta esses filmes no espaço de fatores latentes.

## Como rodar
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest                          # 36 testes
python scripts/comparar.py      # tabelas de resultados e significância
streamlit run app.py            # app interativo
```

A base é baixada automaticamente na primeira execução. Em redes corporativas com inspeção de
HTTPS, o download pode falhar por certificado; nesse caso, baixe o
[zip](https://files.grouplens.org/datasets/movielens/ml-latest-small.zip) e extraia em `data/`.

## Estrutura
```
app.py               # app Streamlit
src/recomendador/
├── data.py          # download e carga da MovieLens
├── split.py         # divisão temporal e remoção de tags do teste
├── metrics.py       # métricas de ranking
├── evaluate.py      # avaliação padronizada (agregada e por usuário)
├── stats.py         # bootstrap e comparações pareadas
└── models/
    ├── base.py      # interface comum (fit / recommend / usuário novo)
    ├── popularity.py
    ├── content.py   # TF-IDF de gêneros e tags
    ├── item_knn.py  # colaborativa item-item com K vizinhos
    └── svd.py       # fatoração de matrizes (PureSVD)
scripts/comparar.py  # experimento completo
tests/               # 36 testes com pytest
```

## Decisões técnicas
- **Interface única (`Recommender`)**: cada modelo implementa `fit()` e `_score()`; a avaliação
  não muda quando entra um modelo novo.
- **Baselines primeiro**: um modelo complexo só se justifica se vencer o aleatório e a
  popularidade.
- **Similaridade em blocos**: a matriz item-item completa ocuparia centenas de MB; calculada em
  blocos e mantendo só os K vizinhos de cada filme, o pico de memória fica bem menor. Um teste
  garante que blocos pequenos e grandes dão exatamente o mesmo resultado.
- **Reprodutibilidade**: sementes fixas no bootstrap, no baseline aleatório e no SVD.
- **Base fora do Git**: a licença do GroupLens não permite redistribuição; o download é
  automático.

## Trabalho futuro
- Repetir o experimento na MovieLens 1M (cerca de 6 mil usuários) para ganhar poder estatístico.
- Modelo híbrido (conteúdo + colaborativo) para usuários e filmes novos (*cold start*).
- Ajuste sistemático de hiperparâmetros com um conjunto de validação separado.
- GitHub Actions rodando os testes a cada push.

## Licença dos dados
F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets: History and Context.*
ACM Transactions on Interactive Intelligent Systems 5, 4. Uso conforme os termos do GroupLens.

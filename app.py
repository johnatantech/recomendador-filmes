"""App de demonstração: marque filmes de que você gosta e compare as recomendações.

Uso local:  streamlit run app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from recomendador.data import load  # noqa: E402
from recomendador.models import (  # noqa: E402
    ItemKNNRecommender, PopularityRecommender, SVDRecommender,
)

REPO = "https://github.com/johnatantech/recomendador-filmes"
K = 10
MODELOS = {
    "Popularidade": {
        "descricao": "Os filmes mais avaliados da base. A lista é igual para todo mundo.",
        "ndcg": "0,040", "cobertura": "1,2%",
    },
    "Item-item": {
        "descricao": "Quem assistiu aos filmes que você marcou também assistiu a estes.",
        "ndcg": "0,055", "cobertura": "7,0%",
    },
    "SVD": {
        "descricao": "Encontra os padrões de gosto escondidos nas notas e projeta o seu perfil neles.",
        "ndcg": "0,069", "cobertura": "6,8%",
    },
}

st.set_page_config(page_title="Recomendador de filmes", page_icon="🎬", layout="wide")


@st.cache_data(show_spinner="Baixando a base MovieLens...")
def carregar_dados() -> tuple[pd.DataFrame, pd.DataFrame]:
    ratings, movies = load()
    avaliacoes = ratings.groupby("movieId").size().rename("avaliacoes")
    movies = movies.set_index("movieId").join(avaliacoes, how="inner")
    movies["generos"] = (
        movies["genres"].str.replace("(no genres listed)", "sem gênero", regex=False)
        .str.replace("|", ", ", regex=False)
    )
    return ratings, movies.sort_values("avaliacoes", ascending=False)


@st.cache_resource(show_spinner="Treinando os modelos (só na primeira vez)...")
def treinar(_ratings: pd.DataFrame) -> dict:
    return {
        "Popularidade": PopularityRecommender("contagem").fit(_ratings),
        "Item-item": ItemKNNRecommender(k_vizinhos=200).fit(_ratings),
        "SVD": SVDRecommender(fatores=50, implicito=False).fit(_ratings),
    }


with st.sidebar:
    st.header("Como funciona")
    st.write(
        "Três abordagens de recomendação recebem os mesmos filmes e devolvem "
        f"{K} sugestões cada. Os números de cada coluna vêm da avaliação offline "
        "do projeto, feita com 542 usuários da MovieLens."
    )
    st.markdown(
        "- **NDCG@10**: qualidade do ranking (quanto maior, melhor).\n"
        "- **Cobertura**: parte do catálogo que o modelo chega a recomendar."
    )
    st.markdown(f"[Código, metodologia e resultados no GitHub]({REPO})")
    st.caption(
        "Dados: MovieLens ml-latest-small (F. M. Harper e J. A. Konstan, 2015), "
        "usados conforme os termos do GroupLens."
    )

st.title("🎬 Recomendador de filmes")
st.write("Marque filmes de que você gosta e compare o que cada modelo sugere.")

try:
    ratings, movies = carregar_dados()
except RuntimeError as erro:
    st.error(f"Não foi possível carregar a base de filmes. {erro}")
    st.stop()

modelos = treinar(ratings)

escolhidos = st.multiselect(
    "Filmes de que você gosta",
    options=movies.index.tolist(),
    format_func=lambda m: movies.at[m, "title"],
    max_selections=20,
    placeholder="Digite o nome de um filme, em inglês. Ex.: Toy Story, Matrix, Amélie",
)

if not escolhidos:
    st.info(
        "Escolha pelo menos um filme para ver as recomendações. "
        "Com 3 a 5 filmes, as diferenças entre os modelos ficam bem visíveis."
    )
    st.stop()

historico = {m: 5.0 for m in escolhidos}
recomendacoes = {nome: modelo.recommend_for_history(historico, K) for nome, modelo in modelos.items()}

colunas = st.columns(len(MODELOS))
for coluna, (nome, info) in zip(colunas, MODELOS.items()):
    with coluna:
        st.subheader(nome)
        st.caption(info["descricao"])
        st.markdown(f"NDCG@10 **{info['ndcg']}**, cobertura **{info['cobertura']}**")
        for posicao, movie_id in enumerate(recomendacoes[nome], start=1):
            st.markdown(f"**{posicao}. {movies.at[movie_id, 'title']}**")
            detalhe = movies.at[movie_id, "generos"]
            if nome == "Item-item":
                origem = modelos[nome].explain(movie_id, historico)
                if origem is not None:
                    detalhe += f"  \nporque você marcou *{movies.at[origem, 'title']}*"
            st.caption(detalhe)

todas = [m for lista in recomendacoes.values() for m in lista]
repetidos = {m for m in todas if todas.count(m) > 1}
st.divider()
st.write(
    f"**{len(repetidos)} de {len(set(todas))} filmes** aparecem em mais de uma lista. "
    "Quanto menos sobreposição, mais cada modelo enxerga o seu gosto de um jeito diferente."
)

"""Download e carga da base MovieLens (ml-latest-small).

A licença do GroupLens não permite redistribuir a base, por isso ela é
baixada sob demanda e a pasta data/ fica fora do Git.
"""
import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def download(dest: Path = DATA_DIR) -> Path:
    target = dest / "ml-latest-small"
    if target.exists():
        return target
    dest.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(URL, timeout=60) as resp:
            zipfile.ZipFile(io.BytesIO(resp.read())).extractall(dest)
    except OSError as exc:
        raise RuntimeError(
            f"Falha ao baixar a MovieLens ({exc}). Baixe manualmente em {URL} "
            f"e extraia em {dest}"
        ) from exc
    return target


def load(dest: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retorna (ratings, movies)."""
    path = download(dest)
    ratings = pd.read_csv(path / "ratings.csv")
    ratings["timestamp"] = pd.to_datetime(ratings["timestamp"], unit="s")
    movies = pd.read_csv(path / "movies.csv")
    return ratings, movies


def load_tags(dest: Path = DATA_DIR) -> pd.DataFrame:
    """Tags livres que os usuários aplicaram aos filmes."""
    tags = pd.read_csv(download(dest) / "tags.csv")
    tags["timestamp"] = pd.to_datetime(tags["timestamp"], unit="s")
    return tags

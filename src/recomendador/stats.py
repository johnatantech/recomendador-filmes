"""Intervalos de confiança por bootstrap.

A pergunta: "se eu tivesse avaliado outro grupo de usuários parecido, o resultado
mudaria?" O bootstrap simula isso sorteando usuários com reposição milhares de vezes
e observando quanto a média varia.

- bootstrap_ci: intervalo de confiança da média de uma métrica.
- paired_bootstrap: compara dois modelos nos MESMOS usuários (pareado). Como cada
  usuário é avaliado pelos dois modelos, a comparação fica bem mais sensível do
  que comparar dois intervalos separados.
"""
import numpy as np
import pandas as pd


def _medias_bootstrap(v: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)


def bootstrap_ci(
    valores, n_boot: int = 2000, confianca: float = 0.95, seed: int = 42
) -> tuple[float, float, float]:
    """Retorna (média, limite inferior, limite superior)."""
    v = np.asarray(valores, dtype=float)
    medias = _medias_bootstrap(v, n_boot, seed)
    alfa = (1 - confianca) / 2
    return float(v.mean()), float(np.quantile(medias, alfa)), float(np.quantile(medias, 1 - alfa))


def paired_bootstrap(
    a: pd.Series, b: pd.Series, n_boot: int = 2000, confianca: float = 0.95, seed: int = 42
) -> dict:
    """Compara a - b usuário a usuário. As séries devem ser indexadas por userId.

    significativo=True quando o intervalo da diferença não contém zero.
    """
    a, b = a.align(b, join="inner")
    d = a.to_numpy(dtype=float) - b.to_numpy(dtype=float)
    medias = _medias_bootstrap(d, n_boot, seed)
    alfa = (1 - confianca) / 2
    inf, sup = float(np.quantile(medias, alfa)), float(np.quantile(medias, 1 - alfa))
    p_valor = float(min(1.0, 2 * min((medias <= 0).mean(), (medias >= 0).mean())))
    return {
        "diferenca": float(d.mean()),
        "ic_inf": inf,
        "ic_sup": sup,
        "p_valor": p_valor,
        "significativo": bool(inf > 0 or sup < 0),
    }

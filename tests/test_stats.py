import numpy as np
import pandas as pd

from recomendador.stats import bootstrap_ci, paired_bootstrap


def test_intervalo_contem_a_media():
    media, inf, sup = bootstrap_ci(np.random.default_rng(0).random(200))
    assert inf <= media <= sup


def test_valores_constantes_tem_intervalo_zero():
    assert bootstrap_ci([0.5] * 50) == (0.5, 0.5, 0.5)


def test_mesmo_resultado_com_mesma_semente():
    v = np.random.default_rng(1).random(100)
    assert bootstrap_ci(v, seed=7) == bootstrap_ci(v, seed=7)


def test_modelos_identicos_empatam():
    a = pd.Series(np.random.default_rng(2).random(300))
    r = paired_bootstrap(a, a.copy())
    assert r["diferenca"] == 0 and not r["significativo"]


def test_diferenca_clara_e_significativa():
    rng = np.random.default_rng(3)
    b = pd.Series(rng.random(300))
    a = b + 0.2 + rng.normal(0, 0.05, 300)
    r = paired_bootstrap(a, b)
    assert r["significativo"] and r["ic_inf"] > 0 and r["p_valor"] < 0.01


def test_pareamento_alinha_por_usuario():
    a = pd.Series([1.0, 0.0], index=[10, 20])
    b = pd.Series([0.0, 1.0], index=[20, 10])  # mesma informação, outra ordem
    assert paired_bootstrap(a, b)["diferenca"] == 0

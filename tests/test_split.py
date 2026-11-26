import pandas as pd

from recomendador.split import temporal_leave_last_out


def _ratings():
    rows = [(u, m, 4.0, pd.Timestamp("2020-01-01") + pd.Timedelta(days=m))
            for u in (1, 2) for m in range(12)]
    rows += [(3, m, 3.0, pd.Timestamp("2020-01-01")) for m in range(3)]  # poucas notas
    return pd.DataFrame(rows, columns=["userId", "movieId", "rating", "timestamp"])


def test_teste_contem_as_ultimas_avaliacoes():
    train, test = temporal_leave_last_out(_ratings(), n_test=5, min_ratings=10)
    for u in (1, 2):
        assert test[test.userId == u]["timestamp"].min() > train[train.userId == u]["timestamp"].max()
        assert len(test[test.userId == u]) == 5


def test_usuarios_com_poucas_notas_sao_removidos():
    train, test = temporal_leave_last_out(_ratings(), min_ratings=10)
    assert 3 not in set(train.userId) | set(test.userId)

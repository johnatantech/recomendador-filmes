import pytest

from recomendador.metrics import ndcg_at_k, precision_at_k, recall_at_k


def test_precision_metade_dos_acertos():
    assert precision_at_k([1, 2, 3, 4], {1, 3}, k=4) == 0.5


def test_recall_todos_relevantes_encontrados():
    assert recall_at_k([1, 2, 3], {1, 3}, k=3) == 1.0


def test_ndcg_ranking_perfeito_vale_um():
    assert ndcg_at_k([1, 2, 9], {1, 2}, k=3) == pytest.approx(1.0)


def test_ndcg_acerto_no_fim_vale_menos_que_no_inicio():
    assert ndcg_at_k([9, 8, 1], {1}, k=3) < ndcg_at_k([1, 8, 9], {1}, k=3)


def test_sem_relevantes_retorna_zero():
    assert recall_at_k([1], set(), k=1) == 0.0
    assert ndcg_at_k([1], set(), k=1) == 0.0

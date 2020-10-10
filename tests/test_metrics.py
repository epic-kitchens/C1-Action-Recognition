import numpy as np

from utils.metrics import topk_accuracy


def test_accuracy_at_1():
    ranks = np.array([[1, 2, 3], [2, 3, 1], [3, 1, 2]])
    labels = np.array([1, 2, 3])

    accuracy = topk_accuracy(ranks, labels, ks=1)[0]

    assert accuracy == 1


def test_accuracy_at_2():
    ranks = np.array([[1, 2, 3], [2, 3, 1], [3, 1, 2], [1, 2, 3]])
    labels = np.array([1, 3, 2, 1])

    accuracy = topk_accuracy(ranks, labels, ks=2)[0]

    assert accuracy == 0.75


def test_accuracy_at_3():
    ranks = np.array([[1, 2, 3], [2, 3, 1], [3, 1, 2], [1, 2, 3]])
    labels = np.array([1, 3, 2, 1])

    accuracy = topk_accuracy(ranks, labels, ks=3)[0]

    assert accuracy == 1


def test_accuracy_at_1_and_3():
    ranks = np.array([[1, 2, 3], [2, 3, 1], [3, 1, 2], [1, 2, 3]])
    labels = np.array([1, 3, 2, 1])

    accuracy = topk_accuracy(ranks, labels, ks=(1, 3, 5))

    assert np.all(accuracy == np.array([0.5, 1, 1]))

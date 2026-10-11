import numpy as np
from src.evaluation import precision_at_k

X = np.array([[1.00, 0.00],
              [0.96, 0.28],
              [0.00, 1.00],
              [0.28, 0.96]])
labels = np.array(["A", "A", "B", "B"])


def test_precision_at_1():
    assert precision_at_k(X, labels, k=1) == 1.0


def test_precision_at_2():
    assert np.isclose(precision_at_k(X, labels, k=2), 0.5)
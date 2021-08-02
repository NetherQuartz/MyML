import pytest
import numpy as np

from functools import reduce

from myml.dl import Tensor


def test_add():
    a = Tensor([[1, 2], [3, 4]])
    b = Tensor([[0, -1], [-1, 0]])
    assert ((a + b).array == a.array + b.array).all()


def test_mul():
    a = Tensor([[1, 2], [3, 4]])
    b = Tensor([[0, -1], [-1, 0]])
    assert ((a * b).array == a.array * b.array).all()


def test_get_array():
    a = Tensor([[1, 2], [3, 4]])
    assert id(a.array) == id(a._array)
    with pytest.raises(AttributeError):
        a.array = np.array([0, 1])


def test_get_shape():
    a = Tensor([[1, 2], [3, 4]])
    assert len(a.shape) == len(a._array.shape) and reduce(lambda t, s: t[0] == t[1], zip(a._array.shape, a.shape))
    with pytest.raises(AttributeError):
        a.shape = np.array([0, 1])


def test_get_requires_grad():
    a = Tensor([[1, 2], [3, 4]], requires_grad=True)
    assert a.requires_grad == a._requires_grad and a.requires_grad is True
    a.requires_grad = False
    assert a.requires_grad == a._requires_grad and a.requires_grad is False


def test_requires_grad_on_results():
    a = Tensor([[1, 2], [3, 4]], requires_grad=False)
    b = Tensor([[0, -1], [-1, 0]], requires_grad=False)
    assert (a + b).requires_grad is False
    a = Tensor([[1, 2], [3, 4]], requires_grad=True)
    b = Tensor([[0, -1], [-1, 0]], requires_grad=False)
    assert (a + b).requires_grad is True
    a = Tensor([[1, 2], [3, 4]], requires_grad=False)
    b = Tensor([[0, -1], [-1, 0]], requires_grad=True)
    assert (a + b).requires_grad is True
    a = Tensor([[1, 2], [3, 4]], requires_grad=True)
    b = Tensor([[0, -1], [-1, 0]], requires_grad=True)
    assert (a + b).requires_grad is True

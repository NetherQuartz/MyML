import pytest
import numpy as np

from functools import reduce

from myml.dl import Tensor


def test_add():
    a = Tensor([[1, 2], [3, 4]])
    b = Tensor([[0, -1], [-1, 0]])
    assert ((a + b).array == a.array + b.array).all()
    assert ((b + 2).array == b.array + 2).all()
    assert ((2 + b).array == b.array + 2).all()


def test_sub():
    a = Tensor([[1, 2], [3, 4]])
    b = Tensor([[0, -1], [-1, 0]])
    assert ((a - b).array == a.array - b.array).all()
    assert ((b - 2).array == b.array - 2).all()
    assert ((2 - b).array == 2 - b.array).all()


def test_mul():
    a = Tensor([[1, 2], [3, 4]])
    b = Tensor([[0, -1], [-1, 0]])
    assert ((a * b).array == a.array * b.array).all()
    assert ((b * 2).array == b.array * 2).all()
    assert ((2 * b).array == b.array * 2).all()


def test_div():
    a = Tensor([[1, 2], [3, 4]])
    b = Tensor([[0, -1], [-1, 0]])
    assert ((a / b).array == a.array / b.array).all()
    assert ((b / 2).array == b.array / 2).all()
    assert ((2 / b).array == 2 / b.array).all()


def test_pow():
    a = Tensor([[1, 2], [3, 4]])
    b = Tensor([[0, -1], [-1, 0]])
    assert ((a ** b).array == a.array ** b.array).all()
    assert ((b ** 2).array == b.array ** 2).all()
    assert ((2 ** b).array == 2 ** b.array).all()


def test_unary():
    a = Tensor([[1, 2], [3, 4]])
    assert ((-a).array == -a.array).all()
    assert ((+a).array == +a.array).all()


def test_get_array():
    a = Tensor([[1, 2], [3, 4]])
    assert id(a.array) == id(a._array)
    with pytest.raises(AttributeError):
        a.array = np.array([0, 1])


def test_get_graph():
    a = Tensor([[1, 2], [3, 4]])
    a *= a
    assert id(a.graph) == id(a._graph)
    with pytest.raises(AttributeError):
        a.graph = None


def test_get_grad():
    pass


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


def test_diff():
    x = Tensor([3], requires_grad=True)
    a = Tensor([4])
    b = Tensor([5])

    y = a * x + b
    y.backward()
    assert (y.grad.array == a.array).all()

    y = a * x ** Tensor([2]) + b * x + b  # y = ax^2 + bx + b
    y.backward()  # y' = 2ax + b
    assert (y.grad.array == np.array([29])).all()

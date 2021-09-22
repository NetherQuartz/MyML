from __future__ import annotations

import numpy as np
from typing import Union
from functools import reduce

from .graph import Node


class Tensor:

    def __init__(self,
                 array: Union[np.ndarray, list],
                 func: str = None,
                 args: list[Node] = None,
                 requires_grad: bool = None,
                 dtype=np.float32):

        if type(array) in [int, float]:
            array = [array]

        self._array: np.ndarray = np.array(array, dtype=dtype)
        self._graph: Node = Node(self, func, args)

        if requires_grad is not None:
            self._requires_grad = requires_grad
        elif len(self._graph.args) > 0:
            self._requires_grad = reduce(lambda t, a: t.tensor.requires_grad or a.tensor.requires_grad,
                                         self._graph.args)
        else:
            self._requires_grad = False

        self._grad: Tensor = None

    @property
    def array(self):
        return self._array

    @property
    def shape(self):
        return self._array.shape

    @property
    def requires_grad(self):
        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, value):
        self._requires_grad = value

    @property
    def grad(self):
        return self._grad

    @property
    def graph(self):
        return self._graph

    def __add__(self, other) -> Tensor:
        if type(other) in [int, float]:
            other = Tensor([other])
        return Tensor(array=self._array + other._array,
                      func="+", args=[self._graph, other._graph])

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other) -> Tensor:
        if type(other) in [int, float]:
            other = Tensor([other])
        return Tensor(array=self._array - other._array,
                      func="-", args=[self._graph, other._graph])

    def __rsub__(self, other):
        return (-self).__add__(other)

    def __mul__(self, other) -> Tensor:
        if type(other) in [int, float]:
            other = Tensor([other])
        return Tensor(array=self._array * other._array,
                      func="*", args=[self._graph, other._graph])

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other) -> Tensor:
        if type(other) in [int, float]:
            other = Tensor([other])
        return Tensor(array=self._array / other._array,
                      func="/", args=[self._graph, other._graph])

    def __rtruediv__(self, other):
        if type(other) in [int, float]:
            other = Tensor([other])
        return Tensor(array=other._array / self._array,
                      func="/", args=[other._graph, self._graph])

    def __pow__(self, other) -> Tensor:
        if type(other) in [int, float]:
            other = Tensor([other])
        return Tensor(array=self._array ** other._array,
                      func="**", args=[self._graph, other._graph])

    def __rpow__(self, other):
        if type(other) in [int, float]:
            other = Tensor([other])
        return Tensor(array=other._array ** self._array,
                      func="**", args=[other._graph, self._graph])

    def __neg__(self):
        return -1 * self

    def __pos__(self):
        return self

    def __str__(self) -> str:
        return str(self._array)

    def __repr__(self):
        str_array = self.array.__str__().split("\n")
        prefix = "Tensor"
        str_array[0] = prefix + str_array[0]
        str_array[1:] = [" " * len(prefix) + line for line in str_array[1:]]
        return "\n".join(str_array)

    @staticmethod
    def _diff(node: Node) -> Tensor:
        if not node.tensor.requires_grad:
            return Tensor([0])

        if len(node.args) == 2:
            a, b = node.args

            if node.func == "+":
                return Tensor._diff(a) + Tensor._diff(b)
            elif node.func == "*":
                return Tensor._diff(a) * b.tensor + Tensor._diff(b) * a.tensor
            elif node.func == "/":
                return (b.tensor * Tensor._diff(a) - Tensor._diff(b) * a.tensor) / (b.tensor ** 2)
            elif node.func == "**":
                if a.tensor.requires_grad and not b.tensor.requires_grad:
                    return b.tensor * a.tensor ** (b.tensor - 1)
                else:
                    raise NotImplementedError
        else:
            return Tensor([1])

    def backward(self):
        self._grad = Tensor._diff(self._graph)

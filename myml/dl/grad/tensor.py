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

    def __add__(self, other: Tensor) -> Tensor:
        return Tensor(array=self._array + other._array,
                      func="+", args=[self._graph, other._graph])

    def __mul__(self, other: Tensor) -> Tensor:
        return Tensor(array=self._array * other._array,
                      func="*", args=[self._graph, other._graph])

    def __pow__(self, other: Tensor) -> Tensor:
        return Tensor(array=self._array ** other._array, func="**", args=[self._graph, other._graph])

    def __str__(self) -> str:
        return str(self._array)

    @staticmethod
    def _diff(node: Node) -> Tensor:
        if not node.tensor.requires_grad:
            return Tensor([0])

        if len(node.args) == 2:
            a, b = node.args

            if node.func == "+":
                if a.tensor.requires_grad and b.tensor.requires_grad:
                    return Tensor._diff(a) + Tensor._diff(b)
                elif not a.tensor.requires_grad and b.tensor.requires_grad:
                    return Tensor._diff(b)
                elif a.tensor.requires_grad and not b.tensor.requires_grad:
                    return Tensor._diff(a)
                else:
                    return Tensor([0])
            elif node.func == "*":
                if a.tensor.requires_grad and b.tensor.requires_grad:
                    return Tensor._diff(a) * b.tensor + Tensor._diff(b) * node.args[
                        0].tensor
                elif not a.tensor.requires_grad and b.tensor.requires_grad:
                    return a.tensor * Tensor._diff(b)
                elif a.tensor.requires_grad and not b.tensor.requires_grad:
                    return Tensor._diff(a) * b.tensor
                else:
                    return Tensor([0])
            elif node.func == "**":
                if a.tensor.requires_grad and not b.tensor.requires_grad:
                    return b.tensor * a.tensor ** (b.tensor + Tensor([-1]))
        else:
            return Tensor([1])

    def backward(self):
        self._grad = Tensor._diff(self._graph)

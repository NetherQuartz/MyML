from __future__ import annotations

import numpy as np
from typing import Union

from .graph import Node


class Tensor:

    def __init__(self,
                 array: Union[np.ndarray, list],
                 func: str = None,
                 args: list = None,
                 requires_grad: bool = False):

        self._array: np.ndarray = np.array(array)
        self._graph: Node = Node(self, func, args)
        self._requires_grad: bool = requires_grad

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

    def __add__(self, other: Tensor) -> Tensor:
        return Tensor(array=self._array + other._array,
                      func="+", args=[self._graph, other._graph],
                      requires_grad=(self._requires_grad or other._requires_grad))

    def __mul__(self, other: Tensor) -> Tensor:
        return Tensor(array=self._array * other._array,
                      func="*", args=[self._graph, other._graph],
                      requires_grad=(self._requires_grad or other._requires_grad))

    def __str__(self) -> str:
        return str(self._array)

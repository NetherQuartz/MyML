from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .tensor import Tensor


class Node:

    def __init__(self, tensor: Tensor, func: str = None, args: list = None):
        self.tensor: Tensor = tensor
        self.func = func
        self.args = args if args is not None else []

    @staticmethod
    def _rec_str(pad, node):
        tab = "  " * pad
        if node.func is not None:
            lines = [tab + node.func]
            lines += [Node._rec_str(pad + 1, arg) for arg in node.args]
            return "\n".join(lines)
        return tab + str(node.tensor.array)

    def __str__(self):
        return Node._rec_str(0, self)

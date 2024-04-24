from typing import List, Tuple
from nanotorch.nn import Layer
from . import Optimizer


__all__ = ["GD"] 


class GD(Optimizer):
    lr : float
    _layers_require_grad: Tuple[Layer]
    def __init__(
            self, layers: List[Layer],
            lr: float         
    ) -> None:
        super(GD, self).__init__(layers)
        self.lr: float = lr 

    def step(self) -> None:
        for layer in self._layers_require_grad:
            layer.parameter -= self.lr * layer.grad
    
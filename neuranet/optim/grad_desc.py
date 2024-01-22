from typing import List
from neuranet.nn import Layer, Activation
from . import Optimizer


__all__ = ["GD"] 


class GD(Optimizer):
    lr : float
    def __init__(
            self, layers: List[Layer],
            lr: float         
    ) -> None:
        super().__init__(layers)
        self.lr = lr 

    def step(self) -> None:
        for layer in self.wActivLayers:
            layer.parameter -= self.lr * layer.grad
    
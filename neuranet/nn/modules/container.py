from ..base import Module, Layer
from neuranet import Tensor
from typing import List

__all__ = ["Sequential"]

class Sequential(Module):

    def __init__(self, layers: List[Layer]):
        super().__init__()

        for layer in layers: setattr(self, layer.__class__.__name__, layer)

    def forward(self, input: Tensor) -> Tensor:
        out = input
        for layer in self.layers():
            out = layer(out)
        return out  
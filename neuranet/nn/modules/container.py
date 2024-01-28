from neuranet.nn.base import Module, Layer
from neuranet import Tensor
from typing import Tuple

__all__ = ["Sequential"]

class Sequential(Module):

    def __init__(self, *layers: Tuple[Layer]) -> None:
        super(Sequential, self).__init__()

        for layer in layers: setattr(self, layer.__class__.__name__, layer)

    def add_layer(self, layer: Layer) -> None:
        setattr(self, layer.__class__.__name__, layer)

    def forward(self, input: Tensor) -> Tensor:
        out = input
        for layer in self.layers():
            out = layer(out)
        return out
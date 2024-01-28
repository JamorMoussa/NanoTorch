from neuranet.nn import Layer, Activation
from typing import List
from abc import ABC, abstractmethod


__all__ = ["Optimizer", ]

class Optimizer(ABC):
     
    layers: List[Layer]

    def __init__(self, layers: List[Layer]) -> None:
        super(Optimizer, self).__init__()
        self.layers: List[Layer] = layers
        self.wActivLayers: List[Layer]  = list(filter(lambda layer: not isinstance(layer, Activation), self.layers))

    def zero_grad(self) -> None:
        for layer in self.wActivLayers: 
            layer.zero_grad() 

    @abstractmethod
    def step(self) -> None:
        ...
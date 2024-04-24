from nanotorch.nn import Layer, Activation
from typing import List, Tuple
from abc import ABC, abstractmethod


__all__ = ["Optimizer", ]

class Optimizer(ABC):
     
    _layers_require_grad: Tuple[Layer]

    def __init__(self, layers: List[Layer]) -> None:
        super(Optimizer, self).__init__()
        self._layers_require_grad: Tuple[Layer]  = tuple(filter(lambda layer: layer.requires_grad , layers))

    def zero_grad(self) -> None:
        for layer in self._layers_require_grad: 
            layer.zero_grad() 

    @abstractmethod
    def step(self) -> None:
        ...
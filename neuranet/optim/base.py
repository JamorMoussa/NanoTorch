from ..nn import Layer
from typing import List
from abc import ABC, abstractmethod


__all__ = ["Optimizer", ]

class Optimizer(ABC):
     
    layers: List[Layer]

    def __init__(self, layers: List[Layer]) -> None:
        self.layers = layers

    def zero_grad(self) ->None:
        for layer in self.layers: layer.zero_grad()

    @abstractmethod
    def step(self):
        ...
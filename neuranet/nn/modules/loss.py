from neuranet import Tensor
from ... import Tensor, sum
from typing import List
from ...nn import Layer, Module
from abc import ABC, abstractmethod

__all__ = ["MSELoss",]

class Loss(Module):

    def __init__(self, layers: List[Layer]) -> None:
        super(Loss, self).__init__()
        self.layers: List[Layer] = layers

    @abstractmethod
    def backward(self, *args, **kwargs) -> None:
        ...



class MSELoss(Loss):
    
    def __init__(self, layers: List[Layer]) ->None:
        super(MSELoss, self).__init__(layers)

    # def forward(self, y_pred: Tensor, y: Tensor) -> Tensor:
    #     return Tensor(sum((y_pred - y)**2))
    
    def forward(self, input: Tensor) -> Tensor:
        ...

    def backward(self, y_pred: Tensor, y: Tensor) -> None:
        grad : Tensor = Tensor((y_pred - y))
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
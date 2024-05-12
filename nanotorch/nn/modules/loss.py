from nanotorch import Tensor, zeros, norm
from nanotorch.nn import Layer, Module
from abc import abstractmethod
from typing import List, Self

__all__ = ["Loss", "MSELoss"]

class Loss(Module):
    
    out_grad: Tensor = None
    _item : Tensor

    def __init__(self, layers: List[Layer]) -> None:
        super(Loss, self).__init__()
        self.layers: List[Layer] = layers
        self._item: Tensor = zeros(1)

    @abstractmethod
    def forward(self, *args, **kwargs) -> Self:
        ...

    def backward(self) -> None:
        out_grad = self.out_grad
        for layer in reversed(self.layers):
            out_grad = layer.backward(out_grad)

    def item(self):
        return self._item



class MSELoss(Loss):
    
    def __init__(self, layers: List[Layer]) ->None:
        super(MSELoss, self).__init__(layers)
    
    def forward(self, y_pred: Tensor, y: Tensor) -> Self:
        dy = (y_pred - y)
        self.out_grad = (2/y.shape[0])*Tensor(dy)
        self._item = norm(Tensor(dy**2))
        return self
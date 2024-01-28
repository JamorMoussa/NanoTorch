from neuranet import Tensor, rand, zeros, dot 
from neuranet.nn.base import Layer
from typing import Tuple

__all__ = ["Linear", ]

class Linear(Layer):

    def __init__(self, *shape: Tuple[int]):
        super().__init__()

        self._parameter: Tensor = rand(*shape)
        self._grad: Tensor = zeros(*shape)

        self.input: Tensor = zeros(*shape)
        self.p_shape: Tuple[int] = shape
        
    def forward(self, input: Tensor) -> Tensor:
        self.input = input
        return dot(input, self._parameter)

    def backward(self, out_grad: Tensor) -> Tensor:
        self.grad = Tensor.dot(self.input.T, out_grad)
        return dot(out_grad, self._parameter.T)
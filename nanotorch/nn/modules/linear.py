from nanotorch import Tensor, rand, zeros, dot 
from nanotorch.nn.base import Layer
from ..parameter import Parameter
from typing import Tuple

__all__ = ["Linear", ]

class Linear(Layer):
    
    requires_grad: bool = True 

    def __init__(self, *shape: Tuple[int]):
        super().__init__()

        self.parameter: Tensor = Parameter(rand(*shape))
        self.grad: Tensor = zeros(*shape)

        self.input: Tensor = zeros(*shape)
        self.p_shape: Tuple[int] = shape
        
    def forward(self, input: Tensor) -> Tensor:
        self.input = input
        return dot(input, self.parameter.data)

    def backward(self, out_grad: Tensor) -> Tensor:
        self.grad = Tensor.dot(self.input.T, out_grad)
        return dot(out_grad, self.parameter.data.T)
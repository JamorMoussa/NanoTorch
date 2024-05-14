from nanotorch import Tensor, rand, zeros, dot 
from nanotorch.nn import Layer
from ..parameter import Parameter
from typing import Tuple

__all__ = ["Linear", ]


class Linear(Layer):
    _parameter: Parameter
    _grad: Tensor
    requires_grad: bool = True 

    def __init__(self, *shape: Tuple[int]):
        super().__init__()

        self.parameter: Tensor = Parameter(rand(*shape))
        self.grad: Tensor = zeros(*shape)

        self.input: Tensor = zeros(*shape)
        self.p_shape: Tuple[int] = shape

    @property
    def parameter(self) -> Parameter:
        return self._parameter
    
    @parameter.setter
    def parameter(self, parameter: Parameter) -> None:
        assert isinstance(parameter, Parameter), "the input is not Parameter type"
        self._parameter = parameter 

    @property
    def grad(self) -> Tensor:
        return self._grad
    
    @grad.setter
    def grad(self, grad: Tensor) -> None:
        self._grad =  grad

    def zero_grad(self):
        self.grad = zeros(*self.parameter.data.shape)
        
    def forward(self, input: Tensor) -> Tensor:
        self.input = input
        return dot(input, self.parameter.data)

    def backward(self, out_grad: Tensor) -> Tensor:
        self.grad = Tensor.dot(self.input.T, out_grad)
        return dot(out_grad, self.parameter.data.T)
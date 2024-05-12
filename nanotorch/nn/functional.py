from nanotorch import Tensor, where
from nanotorch.utils._decorators import input_as
import numpy as np

__all__ = ["relu", "relu_prime", "sigmoid",]

@input_as(DType = Tensor)
def relu(input: Tensor) -> Tensor:
    return where(input > 0, input, 0)

@input_as(DType = Tensor)
def relu_prime(input: Tensor) -> Tensor:
    return where(input > 0 , 1, 0) 

@input_as(DType = Tensor)
def sigmoid(input: Tensor)-> Tensor:
    return Tensor(0.5 * (1 + np.tanh(0.5 * input)))



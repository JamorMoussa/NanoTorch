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
    if np.where(input > 0 ):
        return Tensor(1/(1 + np.exp(-input)))
    else:
        return Tensor(np.exp(input)/(1 + np.exp(input)))



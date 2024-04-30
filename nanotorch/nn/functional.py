from nanotorch import Tensor, where
import numpy as np

__all__ = ["relu", "relu_prime", "sigmoid",]


def relu(input: Tensor) -> Tensor:
    return where(input > 0, input, 0)


def relu_prime(input: Tensor) -> Tensor:
    return where(input > 0 , 1, 0) 


def sigmoid(input: Tensor)-> Tensor:
    return Tensor(1/(1 + np.exp(-input)))


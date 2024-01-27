from .. import Tensor, where

__all__ = ["relu", "lrelu"]


def relu(input: Tensor) -> Tensor:
    return where(input > 0, input, 0)

def lrelu(input: Tensor) -> Tensor:
    return where(input > 0, input, 0.25*input)
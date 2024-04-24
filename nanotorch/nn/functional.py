from nanotorch import Tensor, where

__all__ = ["relu"]


def relu(input: Tensor) -> Tensor:
    return where(input > 0, input, 0)
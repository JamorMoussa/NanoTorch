from nanotorch import Tensor, multiply, tensor2strings, zeros
from abc import ABC, abstractmethod
from typing import List, Callable, Self


__all__ = ["Module", "Layer", "Activation"]
    
class Module(ABC):

    _modules: List[Self] = []

    def __setattr__(self, name, value) -> None:
        super().__setattr__(name, value)
        if isinstance(value, Layer): self.add_module(value)

    def add_module(self, module: Self) -> None: 
        self._modules.append(module)
    
    def layers(self) -> List[Self]:
        return self._modules

    @abstractmethod
    def forward(self, *args, **kwargs) -> Tensor:
        ...

    def backward(self, out_grad: Tensor) -> Tensor:
        ...

    def __call__(self, *args, **kwargs) -> Self:
        return self.forward(*args, **kwargs)


class Layer(Module):

    requires_grad: bool = False 

    def __init__(self) -> None:
        super(Layer, self).__init__()


class Activation(Layer):
    def __init__(self, active_func: Callable[[Tensor], Tensor], active_prime: Callable[[Tensor], Tensor]) -> None:

        self.active_func = active_func
        self.active_prime = active_prime
        
    def forward(self, input: Tensor) -> Tensor:
        self.input = input
        return self.active_func(input)

    def backward(self, out_grad: Tensor) -> Tensor:
        return multiply(out_grad, self.active_prime(self.input))


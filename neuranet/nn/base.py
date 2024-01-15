from .. import Tensor, multiply, tensor2strings
from abc import ABC, abstractmethod
from typing import Dict, List, Callable


__all__ = ["Module", "Layer"]
    
class Module(ABC):

    _modules: Dict[str, 'Module'] = dict()

    def __setattr__(self, name, value) -> None:
        super().__setattr__(name, value)
        if isinstance(value, Layer): self.add_module(name, value)

    
    def add_module(self, name: str, module: 'Module') -> None: 
        self._modules[name] = module
    
    def children(self) -> Dict[str, 'Module']:
        return self._modules

    @abstractmethod
    def forward(self, input: Tensor) -> Tensor:
        ...

    @abstractmethod
    def backward(self, out_grad: Tensor) -> Tensor:
        ...

    def __call__(self, input: Tensor) -> Tensor:
        return self.forward(input)


class Layer(Module):
    _parameters: List[Tensor] 
    grad: Tensor

    def __repr__(self) -> str:
        pram_str = tensor2strings(self._parameters
                                  ).replace('\n ', '\n\t\t')
        
        grad_str = tensor2strings(self.grad
                                  ).replace('\n ', '\n\t\t')
        
        return f"{self.__class__.__name__}(\n\tParamerts:\n\t\t{pram_str}\n\tgrad:\n\t\t{grad_str}\n)"
    

class Activation(Layer):
    def __init__(self, active_func: Callable[[Tensor], Tensor], active_prime: Callable[[Tensor], Tensor]) -> None:

        self.active_func = active_func
        self.active_prime = active_prime
        
    def forward(self, input: Tensor) -> Tensor:
        self.input = input
        return self.active_func(input)

    def backward(self, out_grad: Tensor) -> Tensor:
        return multiply(out_grad, self.active_prime(self.input))


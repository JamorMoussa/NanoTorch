from .. import Tensor
from abc import ABC, abstractmethod
from typing import Dict, Optional, Self


    
class Module(ABC):

    _modules: Dict[str, Optional['Module']] = dict()
    
    def add_module(self, name: str, module: 'Module') -> None: 
        self._modules[name] = module
    
    def children(self):
        return self._modules

    @abstractmethod
    def forward(self, input: Tensor) -> Tensor:
        ...

    @abstractmethod
    def backward(self, out_grad: Tensor) -> Tensor:
        ...

    def __call__(self, input: Tensor) -> Tensor:
        return self.forward(input)
    


class Linear(Module):
    
    def __init__(self):
        super(Linear, self).__init__()

    def forward(self, input: Tensor) -> Tensor:
        return input
    
    def backward(self, out_grad: Tensor) -> Tensor:
        return out_grad


from nanotorch.nn.base import Activation
import nanotorch.nn.functional as F
from nanotorch import where 

__all__ = ["ReLU", "Sigmoid"]


class ReLU(Activation):
    
    def __init__(self) -> None:

        super(ReLU, self).__init__(
            F.relu,
            lambda tensor: where(tensor <= 0, 0, 1) 
        )


class Sigmoid(Activation):
    
    def __init__(self) -> None:

        super(Sigmoid, self).__init__(
            active_func = F.sigmoid,
            active_prime = lambda input: F.sigmoid(input)*(1 - F.sigmoid(input)) 
        )
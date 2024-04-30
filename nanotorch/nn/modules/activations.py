from nanotorch.nn.base import Activation
import nanotorch.nn.functional as F
from nanotorch import where 

__all__ = ["ReLU", "Sigmoid"]


class ReLU(Activation):
    
    def __init__(self) -> None:

        super(ReLU, self).__init__(
            active_func = F.relu,
            active_prime = F.relu_prime 
        )


class Sigmoid(Activation):
    """ 
        Sigmoid activation function.
        
        Attributes:
            active_func (function): The sigmoid activation function.
            active_prime (function): The derivative of the sigmoid activation function.
          
    """
    def __init__(self) -> None:

        super(Sigmoid, self).__init__(
            active_func = F.sigmoid,
            active_prime = lambda input: F.sigmoid(input)*(1 - F.sigmoid(input)) 
        )
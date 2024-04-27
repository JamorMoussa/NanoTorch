from nanotorch.nn.base import Activation
import nanotorch.nn.functional as F
from nanotorch import where 

__all__ = ["ReLU"]


class ReLU(Activation):
    
    def __init__(self) -> None:

        super(ReLU, self).__init__(
            F.relu,
            lambda tensor: where(tensor <= 0, 0, 1) 
        )
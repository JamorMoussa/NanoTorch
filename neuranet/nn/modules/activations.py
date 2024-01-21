from ..base import Activation
from ... import functionnal as F 

__all__ = ["ReLU", ]


class ReLU(Activation):
    
    def __init__(self) -> None:

        super(ReLU, self).__init__(
            F.relu,
            F.relu
        )
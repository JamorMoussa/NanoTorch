from ..base import Activation
from ... import functionnal as F 

__all__ = ["ReLU", "LReLU"]


class ReLU(Activation):
    
    def __init__(self) -> None:

        super(ReLU, self).__init__(
            F.relu,
            F.relu
        )

class LReLU(Activation):

    def __init__(self) -> None:
        super(LReLU, self).__init__(
            F.lrelu,
            F.lrelu
        )
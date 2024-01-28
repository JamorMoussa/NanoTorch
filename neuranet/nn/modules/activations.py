from neuranet.nn.base import Activation
import neuranet.nn.functional as F 

__all__ = ["ReLU"]


class ReLU(Activation):
    
    def __init__(self) -> None:

        super(ReLU, self).__init__(
            F.relu,
            F.relu
        )
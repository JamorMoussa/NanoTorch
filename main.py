import neuranet as nnt
from neuranet import Tensor
import neuranet.nn as nn 


class MLP(nn.Module):

    def __init__(self):

        self.l1 = nn.Linear(3, 2)
        self.l2 = nn.Linear(2, 3)
        self.l3 = nn.Linear(3, 5)

    def forward(self, input: Tensor) -> Tensor:
        out = self.l1(input)
        out = self.l2(out)
        return self.l3(out)
    
    
model = MLP()


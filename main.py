import neuranet as nnt
import neuranet.nn as nn 

class NewClass(nn.Module):

    def __init__(self):
        super().__init__()

        self.l1 = nn.Linear()
        self.l2 = nn.Linear()

    def forward(self, input: nnt.Tensor) -> nnt.Tensor:
        return input
    
    def backward(self, out_grad: nnt.Tensor) -> nnt.Tensor:
        return out_grad
    

n = NewClass()
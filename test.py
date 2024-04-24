import nanotorch as nnt 
import nanotorch.nn.functional as F

print(F.relu(nnt.Tensor([1, -1, 2])))
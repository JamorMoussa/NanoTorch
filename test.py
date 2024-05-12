import nanotorch as nnt
import nanotorch.nn as nn


param = nn.Parameter(nnt.rand(4, 3))

print(param)
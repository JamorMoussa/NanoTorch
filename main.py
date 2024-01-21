import neuranet as nnt
import neuranet.nn as nn
import numpy as np  


class MLP(nn.Module):

    def __init__(self):

        self.l1 = nn.Linear(1, 5)
        self.a1 = nn.ReLU()
        self.l2 = nn.Linear(5, 3)
        self.a2 = nn.ReLU()
        self.l3 = nn.Linear(3, 1)

    def forward(self, input: nnt.Tensor) -> nnt.Tensor:
        out = self.l1(input)
        out = self.a1(out)
        out = self.l2(out)
        out = self.a2(out)
        return self.l3(out)
    

X_train = nnt.rand(100, 1)

#y = nnt.dot(X_train, nnt.Tensor([1, -2, 3]).T)  

y = nnt.Tensor(X_train**2 + 1)  
    
model = MLP()

mse = nn.MSELoss(model.layers())

opt = nnt.optim.GD(model.layers(), lr=0.01)


for epoch in range(100):

    for xi, yi in zip(X_train, y):

        opt.zero_grad()

        y_predi = model(nnt.Tensor(xi))

        loss = mse(y_predi, nnt.Tensor(yi))

        loss.backward()

        opt.step()


layers = model.layers()

print(model(2))

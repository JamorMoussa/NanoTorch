import neuranet as nnt
import neuranet.nn as nn 


class MLP(nn.Module):

    def __init__(self):

        self.l1 = nn.Linear(3, 1)

    def forward(self, input: nnt.Tensor) -> nnt.Tensor:
        return self.l1(input)
    

X_train = nnt.rand(100, 3)

y = nnt.dot(X_train, nnt.Tensor([1, -2, 3]).T)    
    
model = MLP()

mse = nn.MSELoss(model.layers())

opt = nnt.optim.GD(model.layers(), lr=0.001)


for epoch in range(3000):

    for xi, yi in zip(X_train, y):

        opt.zero_grad()

        y_predi = model(nnt.Tensor(xi))

        loss = mse(y_predi, nnt.Tensor(yi))

        loss.backward()

        opt.step()


print(model.layers()[0].parameter)


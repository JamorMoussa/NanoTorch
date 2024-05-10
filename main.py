import nanotorch as nnt
import nanotorch.nn as nn
import nanotorch.optim as optim


class MLPModel(nn.Module):

    def __init__(self):

        self.fc = nn.Sequential(
            nn.Linear(3, 3),
            nn.Sigmoid(),
            nn.Linear(3, 5),
            nn.Sigmoid(), 
            nn.Linear(5, 1)
        )

    def forward(self, input: nnt.Tensor) -> nnt.Tensor:
        return self.fc(input)
    

X = nnt.rand(100, 3)
y = nnt.dot(X, nnt.Tensor([1, 2, 3]).T) 



model = MLPModel()

mse = nn.MSELoss(model.layers())

opt = optim.SGD(model.layers(), lr=0.1)

for epoch in range(100):

    for xi, yi in zip(X, y):

        opt.zero_grad()

        y_predi = model(nnt.Tensor(xi))

        loss = mse(y_predi, nnt.Tensor(yi))

        loss.backward()

        opt.step()

print(model(1))

# output: 

# [[1.00772484]
#  [1.98651816]
#  [3.04503581]]
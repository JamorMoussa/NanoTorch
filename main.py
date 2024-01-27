import neuranet as nnt
import neuranet.nn as nn


# class MLP(nn.Module):

#     def __init__(self):

#         self.l1 = nn.Linear(1, 5)
#         self.a1 = nn.ReLU()
#         self.l2 = nn.Linear(5, 3)
#         self.a2 = nn.ReLU()
#         self.l3 = nn.Linear(3, 1)

#     def forward(self, input: nnt.Tensor) -> nnt.Tensor:
#         out = self.l1(input)
#         out = self.a1(out)
#         out = self.l2(out)
#         out = self.a2(out)
#         return self.l3(out)


    

X_train = nnt.rand(1000, 1)

# y = nnt.dot(X_train, nnt.Tensor([[1, -2, 3],
#                                 [1, 1, 1]]).T)  

y = nnt.Tensor(nnt.dot(X_train, nnt.Tensor([2])) + 1)

model = nn.Sequential([
    nn.Linear(1, 2),
    nn.ReLU(),
    nn.Linear(2, 3),
    nn.ReLU(),
    nn.Linear(3, 2),
    nn.ReLU(),
    nn.Linear(2, 1)
])

mse = nn.MSELoss(model.layers())

opt = nnt.optim.GD(model.layers(), lr=0.01)

for epoch in range(10):
    
    #for xi, yi in zip(X_train, y):
    
    for i in range(0, 1000, 50):

        opt.zero_grad()

        y_predi = model(nnt.Tensor(X_train[i:i+50]))

        loss = mse(y_predi, nnt.Tensor(y[i:i+50]))

        if i%1000 == 0:
            print(loss.items())

        loss.backward()

        opt.step()


layers = model.layers()

print(model(2))

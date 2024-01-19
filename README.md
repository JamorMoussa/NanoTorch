# NeuraNet

**NeuraNet** is a deep learning library (micro-framework) inspired by the PyTorch framework, which 
I created using only **Math** and **Numpy** :). My purpose here is not to create a powerful deep 
learning framework (maybe in the future), but solely to understand how deep learning frameworks like PyTorch and TensorFlow work behind the scenes.

## Neural Networks:

Let's explore an example of building a simple neural network (essentially a Linear Regression model) with **NeuraNet**:

```python
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

for epoch in range(30):

    for xi, yi in zip(X_train, y):

        opt.zero_grad()

        y_predi = model(nnt.Tensor(xi))

        loss = mse(y_predi, nnt.Tensor(yi))

        loss.backward()

        opt.step()

print(model.layers()[0].parameter)
```

The output is as follows:

```
[[ 0.99887237]
 [-1.99063777]
 [ 2.99119258]]
```
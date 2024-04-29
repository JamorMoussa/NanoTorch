# NanoTorch

**NanoTorch** is a deep learning library (micro-framework) inspired by the PyTorch framework, which 
I created using only **Math** and **Numpy** :). My purpose here is not to create a powerful deep 
learning framework (maybe in the future), but solely to understand how deep learning frameworks like PyTorch and TensorFlow work behind the scenes.

## New Features in this Update:

Explore the latest enhancements and additions in this update by checking the new features documentation. [new features documentation](new_features.md).

## Neural Networks:

Let's explore an example of building a simple neural network (essentially a Linear Regression model) with **NanoTorch**:

```python
import nanotorch as nnt
import nanotorch.nn as nn 
```

Let's build a simple model:

```python
class MLPModel(nn.Module):

    def __init__(self):

        self.fc = nn.Sequential(
            nn.Linear(3, 2),
            nn.ReLU(),
            nn.Linear(2, 1)
        )

    def forward(self, input: nnt.Tensor) -> nnt.Tensor:
        return self.fc(input)
```
Let's generate a simple dataset, using the `nn.rand` function and the `nnt.dot` operation:

```python
X_train = nnt.rand(100, 3)
y = nnt.dot(X_train, nnt.Tensor([1, -2, 3]).T)    
```

Now, let's create an instance of `MlpModel`
```python
model = MLPModel()
```

We are dealing with regression task. So, the `nn.MSELoss` is chosen

```python
mse = nn.MSELoss(model.layers())
```

Let's define the gradient descent optimizer

```python
opt = nnt.optim.GD(model.layers(), lr=0.001)
```

Finally, The training loop

```python
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
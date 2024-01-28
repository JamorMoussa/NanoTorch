# New Features in this Update:

In this latest update, several new features have been introduced to enhance the functionality of the neuranet library:

- `Sequential`: Now you can conveniently construct neural networks using the Sequential module from the nn package.

- `ReLU`: The Rectified Linear Unit (ReLU) activation function is now available for use in your neural network models

## Build models using Sequential :
When utilizing the Sequential module in the neuranet library, there are two approaches you can employ to construct neural networks.

### First Method: Using the Constructor

```python
import neuranet as nnt
import neuranet.nn as nn

model = nn.Sequential(
    nn.Linear(2, 3),
    nn.ReLU(),
    nn.Linear(3, 1)
)
```

### Second Method: Adding Layers

```python
import neuranet as nnt
import neuranet.nn as nn

model = nn.Sequential()

model.add_layer(nn.Linear(2, 3))
model.add_layer(nn.ReLU())
model.add_layer(nn.Linear(3, 1))

```
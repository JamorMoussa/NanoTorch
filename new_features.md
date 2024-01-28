
## Build models using Sequential :
There are two approaches to construct neural networks using the Sequential module in the neuranet library.

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
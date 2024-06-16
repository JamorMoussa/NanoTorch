# Tensor


## What's Tensor?

Deep learning at a low level can be seen as just **tensor** manipulation. A **tensor** can be defined as a generalization of vectors and matrices to higher dimensions. Thus, **tensors** are at the heart of deep learning.

<figure markdown>
    <center>
    <img src= "/images/docs/tensor/tensor.png" width= "500" />
    <figcaption> <b> Tensor </b> </figcaption>
</figure>

The first thing we need to address in building `NanoTorch` is having a powerful and efficient tensors module that ensures numerical stability. Achieving this can be challenging and is not our primary mission here. Therefore, we're going to use `Numpy`, a Python library that provides powerful N-dimensional arrays. `Numpy` is fast (C implementation) and easy to use, making it an excellent choice for handling tensor operations.

However, we need to make some decisions about the way of integrating the `Numpy` library to ensure compatibility with other modules in NanoTorch. We'll discuss these decisions in a few moments.


## Numpy for Numerical Operations

**Numpy** is a library based on `ndarray`, a multi-dimensional array of the same type. It offers algebraic operations in an efficient way. Let's look at some examples with Numpy and then explore how we can use it to build the `tensor` module for `NanoTorch`.



```python
import numpy as np
```

```
├── nanotorch
│   ├── tensor
│   │   ├── __init__.py
│   │   ├── ops.py
│   │   └── tensor.py
```
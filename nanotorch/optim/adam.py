from typing import List, Tuple
from nanonp.nn import Layer
from . import Optimizer
import numpy as np


__all__ = ["Adam", "NAdam", "AMSGrad"] 


### Adaptive Moment Estimation (Adam)
class Adam(Optimizer):
    lr : float
    _layers_require_grad: Tuple[Layer]
    def __init__(
            self, 
            layers: List[Layer],
            lr: float,
            beta1: float,
            beta2: float,
            epsilon: float,       
    ) -> None:
        super(Adam, self).__init__(layers)
        self.lr: float = lr 
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = [np.zeros_like(layer.parameter) for layer in self._layers_require_grad]  
        self.u = [np.zeros_like(layer.parameter) for layer in self._layers_require_grad]  
        self.t = 0

    def step(self):
        self.t += 1
        for i, layer in enumerate(self._layers_require_grad):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * layer.grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (layer.grad ** 2)

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            layer.parameter.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)



### Nesterov-accelerated Adam (NADAM)
class NAdam(Optimizer):
    lr : float
    _layers_require_grad: Tuple[Layer]
    def __init__(self, 
                layers: List[Layer],
                lr: float=0.002,
                beta1: float=0.9,
                beta2: float=0.999,
                epsilon: float=1e-8
                ) -> None:
        super(NAdam, self).__init__(layers)
        self.lr: float = lr 
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = [np.zeros_like(layer.parameter) for layer in self._layers_require_grad]  
        self.u = [np.zeros_like(layer.parameter) for layer in self._layers_require_grad]  
        self.t = 0

    
    def step(self):
        self.t += 1
        for i, layer in enumerate(self._layers_require_grad):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * layer.grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (layer.grad ** 2)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            layer.parameter.data -= self.lr * (self.beta1 * m_hat + (1 - self.beta1) * layer.grad / (1 - self.beta1 ** self.t)) / (np.sqrt(v_hat) + self.epsilon)



### ADAM with Maximum A Posteriori (AMSGrad)
class AMSGrad:
    lr : float
    _layers_require_grad: Tuple[Layer]
    def __init__(self, 
                layers: List[Layer],
                lr: float=0.002,
                beta1: float=0.9,
                beta2: float=0.999,
                epsilon: float=1e-8
                ) -> None:
        super(AMSGrad, self).__init__(layers)
        self.lr: float = lr 
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = [np.zeros_like(layer.parameter) for layer in self._layers_require_grad]  # First moment estimate
        self.u = [np.zeros_like(layer.parameter) for layer in self._layers_require_grad]  #  Second moment estimate
        self.v_hat = [np.zeros_like(layer.parameter) for layer in self._layers_require_grad]  # Maintains the maximum of all v's

        
    def step(self):
        for i, layer in enumerate(self._layers_require_grad):
                # Update biased first moment estimate
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * layer.grad
                # Update biased second raw moment estimate
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (layer.grad ** 2)
                # Maintain the maximum of all second raw moment estimates
                self.v_hat[i] = np.max(self.v_hat[i], self.v[i])
                # Compute the effective learning rate
                step_size = self.lr / (np.sqrt(self.v_hat[i]) + self.epsilon)
                # Update parameters
                layer.parameter.data -= step_size * self.m[i]





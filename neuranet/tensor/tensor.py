import numpy as np 
from typing import Self


class Tensor(np.ndarray):
    """
        Tensor: class represents multi-dimension array in NeuraNet library, It's extends for the numpy 
                ndarray base class, which means that has all features supported by the numpy.adarray. 
                but will include new features in the feature release like 'autograd'.
    """
    def __new__(cls, input: Self) -> Self:
        """
            This class method will create a Tensor.
            
            Args:
                input: is Tensor, list, tuple, np.ndarray. These are the only types supported in the current version.
            
            Returns:
                Returns a Tensor object.
            
            Note: 
                Every Tensor has at least two dimensions, even if the input is 1-dimensional.
                
            Example:
                >>> import neuranet as nnt
                >>> a = nnt.Tensor([1, 2, 3])
                >>> a
                Tensor([[1, 2, 3]]) 
                >>> a.shape 
                (1, 3)
                >>> a.T
                Tensor([[1],
                        [2],
                        [3]])
        """
        
        if not isinstance(input, (Tensor, list, tuple, np.ndarray)): 
            raise ValueError(f"the 'input' attributes must be list, tuple, numpy.ndarray. But '{input.__class__.__name__}' is given") 
        
        # reshape to 2-d if the input is 1-d:
        if not isinstance(input, np.ndarray): input = np.array(input)
        if input.ndim == 1: input = input.reshape(1, -1)
        
        # create a view : 
        obj = np.asanyarray(input).view(cls)

        return obj

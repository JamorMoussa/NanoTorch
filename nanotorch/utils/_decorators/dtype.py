from nanotorch import Tensor
import functools

__all__ = ["input_as", ]

def input_as(DType):

    def check_for_dtype(func):

        @functools.wraps(func)
        def wrapper(input):

            if not isinstance(input, DType):
                raise TypeError(f"The input type must be '{DType.__name__}', not '{input.__class__.__name__}'.")
            
            return func(input)
        
        return wrapper
    return check_for_dtype


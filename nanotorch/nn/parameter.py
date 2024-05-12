from nanotorch import Tensor


__all__ = ["Parameter", ]

class Parameter:

    data: Tensor = None
    requires_grad: bool = True 


    def __init__(
        self,
        data: Tensor = None,
        requires_grad: bool = True
    ) -> None:
    
        self.data = data
        self.requires_grad = requires_grad

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(requires_grad = {self.requires_grad})"
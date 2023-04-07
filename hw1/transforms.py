import numpy as np
import torch


class TensorView(object):
    """
    A transform that returns a new view of a tensor.
    """

    def __init__(self, *view_dims):
        self.view_dims = view_dims

    def __call__(self, tensor: torch.Tensor):
        rensor_args = 1
        for i in range(len(tensor.size())):
            rensor_args *= tensor.size()[i]
        flattened_tensor = tensor.resize(rensor_args,)
        return flattened_tensor


class BiasTrick(object):
    """
    A transform that applies the "bias trick": Adds an element equal to 1 to
    a given tensor.
    """

    def __call__(self, tensor: torch.Tensor):
        assert tensor.dim() == 1, "Only 1-d tensors supported"
        temp = torch.tensor([1])
        torch.cat((tensor,temp))



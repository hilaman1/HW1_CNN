import numpy as np
import torch


class TensorView(object):
    """
    A transform that returns a new view of a tensor.
    """

    def __init__(self, *view_dims):
        self.view_dims = view_dims

    def __call__(self, tensor: torch.Tensor):
        # Use Tensor.view() to implement the transform.
        return tensor.view(self.view_dims)


class BiasTrick(object):
    """
    A transform that applies the "bias trick": Adds an element equal to 1 to
    a given tensor.
    """

    def __call__(self, tensor: torch.Tensor):
        assert tensor.dim() == 1, "Only 1-d tensors supported"
        # Add a 1 at the end of the given tensor.
        # Make sure to use the same data type.
        torch.cat((tensor, torch.tensor([1])))

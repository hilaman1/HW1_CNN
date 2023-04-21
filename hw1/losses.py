import abc
import torch


class ClassifierLoss(abc.ABC):
    """
    Represents a loss function of a classifier.
    """

    def __call__(self, *args, **kwargs):
        return self.loss(*args, **kwargs)

    @abc.abstractmethod
    def loss(self, *args, **kw):
        pass

    @abc.abstractmethod
    def grad(self):
        """
        :return: Gradient of the last calculated loss w.r.t. model
            parameters, as a Tensor of shape (D, C).
        """
        pass


class SVMHingeLoss(ClassifierLoss):
    def __init__(self, delta=1.0):
        self.delta = delta
        self.grad_ctx = {}

    def loss(self, x, y, x_scores, y_predicted):
        """
        Calculates the loss for a batch of samples.

        :param x: Batch of samples in a Tensor of shape (N, D).
        :param y: Ground-truth labels for these samples: (N,)
        :param x_scores: The predicted class score for each sample: (N, C).
        :param y_predicted: The predicted class label for each sample: (N,).
        :return: The classification loss as a Tensor of shape (1,).
        """

        assert x_scores.shape[0] == y.shape[0]
        assert y.dim() == 1

        # TODO: Implement SVM loss calculation based on the hinge-loss formula.
        #
        # Notes:
        # - Use only basic pytorch tensor operations, no external code.
        # - Partial credit will be given for an implementation with only one
        #   explicit loop.
        # - Full credit will be given for a fully vectorized implementation
        #   (zero explicit loops).
        #   Hint: Create a matrix M where M[i,j] is the margin-loss
        #   for sample i and class j (i.e. s_j - s_{y_i} + delta).

        loss = None
        # ====== YOUR CODE: ======
        # x_scores[i,j] = W_i * x_j
        #  -> s_j - s_{y_i} = x_scores[i,j] - x_scores[i,y_i]
        M = x_scores - x_scores.gather(1, y.view(-1, 1)) + self.delta
        L_mat = torch.max(torch.zeros_like(M), M)
        loss = (torch.sum(L_mat) - self.delta * x_scores.shape[0]) / torch.tensor(y.shape[0],dtype=float)
        loss = torch.reshape(loss,(1,))

        # ========================
        # TODO: Save what you need for gradient calculation in self.grad_ctx
        # ====== YOUR CODE: ======
        self.grad_ctx['x'] = x
        self.grad_ctx['y'] = y
        self.grad_ctx['M'] = M
        # ========================
        return loss

    def grad(self):

        # TODO: Implement SVM loss gradient calculation
        # Same notes as above. Hint: Use the matrix M from above, based on
        # it create a matrix G such that X^T * G is the gradient.

        grad = None
        # ====== YOUR CODE: ======

        # when $j!=y_i$, we can write the gradient as:
        # -1/N * \sum_(i=1) (\sum_{j\neq y_i} \mathbb{1} (m_{i,j}>0))x_i + \gamma w_(y_i)
        # when $j!=y_i$, we can write the gradient as 0
        # we get:
        M = self.grad_ctx['M']
        y = self.grad_ctx['y']
        x = self.grad_ctx['x']
        M[M <= 0] = 0.0
        M[M > 0] = 1.0
        M[torch.arange(y.shape[0]), y] = -1 * torch.sum(M, dim=1)
        grad = (x.T @ M) / x.shape[0]

        # M = self.grad_ctx['M']
        # y = self.grad_ctx['y']
        # x = self.grad_ctx['x']
        # ones_zeros_M = torch.where(M > 0, torch.tensor(1.0), torch.tensor(0.0))
        # ones_zeros_M[torch.arange(y.shape[0]), y] = -1 * torch.sum(ones_zeros_M, dim=1)
        # grad = (x.T @ M) / x.shape[0]
        # no regularization because we'll implement it later


        # ========================

        return grad

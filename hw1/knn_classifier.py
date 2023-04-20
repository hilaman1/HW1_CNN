from typing import Sequence, Tuple, List

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

import helpers.dataloader_utils as dataloader_utils
from . import dataloaders

import hw1.dataloaders as hw1dataloaders


class KNNClassifier:
    def __init__(self, k: int):
        self.k = k
        self.x_train = None
        self.y_train = None
        self.n_classes = None

    def train(self, dl_train: DataLoader):
        """
        Trains the KNN model. KNN training is memorizing the training data.
        Or, equivalently, the model parameters are the training data itself.
        :param dl_train: A DataLoader with labeled training sample (should
            return tuples).
        :return: self
        """

        x_train, y_train = dataloader_utils.flatten(dl_train)
        self.x_train = x_train
        self.y_train = y_train
        self.n_classes = len(set(y_train.numpy()))
        return self

    def predict(self, x_test: Tensor) -> Tensor:
        """
        Predict the most likely class for each sample in a given tensor.
        :param x_test: Tensor of shape (N,D) where N is the number of samples.
        :return: A tensor of shape (N,) containing the predicted classes.
        """

        # Calculate distances between training and test samples
        dist_matrix = self.calc_distances(x_test)

        # Implement k-NN class prediction based on distance matrix.
        # For each training sample we'll look for it's k-nearest neighbors.
        # Then we'll predict the label of that sample to be the majority
        # label of it's nearest neighbors.

        n_test = x_test.shape[0]
        y_pred = torch.zeros(n_test, dtype=torch.int64)

        for i in range(n_test):
            # - Find indices of k-nearest neighbors of test sample i
            # - Set y_pred[i] to the most common class among them
            # ====== YOUR CODE: ======
            dists_to_x_test_i = dist_matrix[:, i]  # the i-th column matches the distances to test sample i
            knn_indexes = torch.argsort(dists_to_x_test_i, descending=False)[: self.k]  # indexes matching the k nearest training samples
            knn_labels = torch.gather(self.y_train, dim=-1, index=knn_indexes)  # the labels of the k nearest training samples
            y_pred[i] = torch.argmax(torch.bincount(knn_labels))  # counts how many in each label, and takes the label with maximum counts
            # ========================

        return y_pred

    def calc_distances(self, x_test: Tensor) -> Tensor:
        """
        Calculates the L2 distance between each point in the given test
        samples to each point in the training samples.
        :param x_test: Test samples. Should be a tensor of shape (Ntest,D).
        :return: A distance matrix of shape (Ntrain,Ntest) where Ntrain is the
            number of training samples. The entry i, j represents the distance
            between training sample i and test sample j.
        """

        # Implement L2-distance calculation as efficiently as possible.
        # Notes:
        # - Use only basic pytorch tensor operations, no external code.
        # - No credit will be given for an implementation with two explicit
        #   loops.
        # - Partial credit will be given for an implementation with only one
        #   explicit loop.
        # - Full credit will be given for a fully vectorized implementation
        #   (zero explicit loops).
        # Hint 1: Open the expression (a-b)^2.
        # Hint 2: Use "Broadcasting Semantics".

        dists = torch.tensor([])
        # ====== YOUR CODE: ======
        x_train_squared_norms = torch.sum(self.x_train * self.x_train, dim=1, keepdim=True)  # i-th row contains the norm of i-th train vector
        x_test_squared_norms = torch.sum(x_test * x_test, dim=1, keepdim=True)  # j-th row contains the norm of j-th test vector
        x_train_dot_x_test = torch.matmul(self.x_train, torch.transpose(x_test, 0, 1))  # the i,j entry contains the dot product of the i-th train vector with the j-th test vector
        dists = torch.sqrt(x_train_squared_norms + torch.transpose(x_test_squared_norms, 0, 1) - 2 * x_train_dot_x_test)  # note here that the dimensions are aligned right due to broadcasting
        # ========================
        return dists


def accuracy(y: Tensor, y_pred: Tensor) -> float:
    """
    Calculate prediction accuracy: the fraction of predictions in that are
    equal to the ground truth.
    :param y: Ground truth tensor of shape (N,)
    :param y_pred: Predictions vector of shape (N,)
    :return: The prediction accuracy as a fraction. (between 0 and 1)
    """
    assert y.shape == y_pred.shape
    assert y.dim() == 1

    # Calculate prediction accuracy. Don't use an explicit loop.
    accuracy = None

    # ====== YOUR CODE: ======
    accuracy = float(torch.eq(y_pred, y).sum()) / y.shape[0]
    # ========================

    return accuracy


def find_best_k(ds_train: Dataset, k_choices: Sequence[int], num_folds: int) -> Tuple[int, List[List[float]]]:
    """
    Use cross validation to find the best K for the kNN model.

    :param ds_train: Training dataset.
    :param k_choices: A sequence of possible value of k for the kNN model.
    :param num_folds: Number of folds for cross-validation.
    :return: tuple (best_k, accuracies) where:
        best_k: the value of k with the highest mean accuracy across folds
        accuracies: The accuracies per fold for each k (list of lists).
    """

    accuracies = []

    for i, k in enumerate(k_choices):
        model = KNNClassifier(k)

        # Train model num_folds times with different train/val data.
        # Don't use any third-party libraries.
        # You can use your train/validation splitter from part 1 (even if
        # that means that it's not really k-fold CV since it will be a
        # different split each iteration), or implement something else.

        # ====== YOUR CODE: ======
        accuracy_k = [0.0 for i in range(num_folds)]
        validation_ratio = 1.0 / num_folds

        for fold_num in range(num_folds):
            dl_train, dl_valid = hw1dataloaders.create_train_validation_loaders(ds_train, validation_ratio)
            x_valid, y_valid = dataloader_utils.flatten(dl_valid)
            model.train(dl_train)
            y_pred = model.predict(x_valid)
            accuracy_k[fold_num] = accuracy(y_valid, y_pred)

        accuracies.append(accuracy_k)
        # ========================

    best_k_idx = np.argmax([np.mean(acc) for acc in accuracies])
    best_k = k_choices[best_k_idx]

    return best_k, accuracies

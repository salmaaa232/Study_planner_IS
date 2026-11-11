"""A simple, effort-independent average-mark baseline."""
import numpy as np


class AverageMark:
    def __init__(self, training_mean):
        self.training_mean = float(training_mean)

    def predict(self, x):
        x = np.atleast_2d(x)
        # Use this student's earlier mean when available; otherwise train mean.
        return np.where(x[:, 14] > .5, self.training_mean, x[:, 12]).clip(0, 100)

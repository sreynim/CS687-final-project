import numpy as np


def softmax(self, nums):
    numerator = np.exp(nums)
    return numerator / np.sum(numerator)
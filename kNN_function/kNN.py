from math import sqrt
import numpy as np
from collections import Counter


def test_params(k, train_X, train_y, x):
    assert type(k) is int, "k must be type int"
    assert (type(train_X) is np.ndarray) and (type(train_y) is np.ndarray) and (type(x) is np.ndarray), \
        "train_X, train_y, x must be type numpy.ndarray"
    assert (train_X.ndim == 2) and (train_y.ndim == 1) and (x.ndim == 1), \
        "train_X, train_y, or x must have dimensions of 2, 1, and 1"
    assert 0 < k <= train_X.shape[0], "k must be > 0 and <= # of labeled examples"
    assert train_X.shape[0] == train_y.shape[0], "# of examples must = # of labels"
    assert train_X.shape[1] == x.shape[0], "Number of features in train_X must match that in x"


def kNN_classify(k, train_X, train_y, x):
    # Parameters validation
    test_params(k, train_X, train_y, x)

    # Calculate the distances from the test feature x to labeled features X
    distances = np.array([sqrt(np.sum((train_X - x)**2)) for train_X in train_X])

    # get the list of indices of which
    k_nearest_indices = (np.argsort(distances))[:k]
    votes = Counter([train_y[idx] for idx in k_nearest_indices])
    return votes.most_common(1)[0][0]
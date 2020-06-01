import numpy as np


def train_test_split(X, y, test_ratio=0.2, seed=None):

    assert type(X) is np.ndarray and type(y) is np.ndarray, "X and y must be of type numpy.ndarray"
    assert X.ndim == 2 and y.ndim == 1, "X and y must have dimensions of 2 and 1, respectively"
    assert X.shape[0] == y.shape[0], "Number of examples must match number of labels"
    assert type(test_ratio) is float and 0.0 <= test_ratio <= 1.0, "Test ratio should be between 0.0 and 1.0"
    if seed:
        np.random.seed(seed)

    shuffled_indices = np.random.permutation(X.shape[0])
    test_size = int(len(y) * test_ratio)
    train_size = len(y) - test_size

    test_X = np.array([X[idx] for idx in shuffled_indices[:test_size]])
    test_y = np.array([y[idx] for idx in shuffled_indices[:test_size]])
    train_X = np.array([X[idx] for idx in shuffled_indices[test_size:]])
    train_y = np.array([y[idx] for idx in shuffled_indices[test_size:]])

    return {"train_X": train_X, "train_y": train_y, "test_X": test_X, "test_y": test_y}

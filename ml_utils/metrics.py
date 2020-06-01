import numpy as np


def accuracy_score(y_true, y_predict):

    assert type(y_true) is np.ndarray and type(y_predict) is np.ndarray, \
        "y_true and y_predict must be of type numpy.ndarray"
    assert y_true.ndim == 1 and y_predict.ndim == 1, "y_true and y_predict must have dimensionality of 1"
    assert len(y_true) == len(y_predict), "y_true and y_predict must have same length"

    correct = np.count_nonzero((y_true == y_predict))
    total = y_true.shape[0]
    return correct / total


def mean_squared_error(y_true, y_predict):
    assert type(y_true) is np.ndarray and type(y_predict) is np.ndarray, \
        "y_true and y_predict must be of type numpy.ndarray"
    assert y_true.ndim == 1 and y_predict.ndim == 1 and len(y_true) == len(y_predict), \
        "y_true and y_predict must have dimensionality of 1, and they must have same length"

    return np.sum((y_true - y_predict)**2) / len(y_true)


def root_mean_squared_error(y_true, y_predict):
    from math import sqrt
    return sqrt(mean_squared_error(y_true, y_predict))


def mean_absolute_error(y_true, y_predict):
    assert type(y_true) is np.ndarray and type(y_predict) is np.ndarray, \
        "y_true and y_predict must be of type numpy.ndarray"
    assert y_true.ndim == 1 and y_predict.ndim == 1, "y_true and y_predict must have dimensionality of 1"
    assert len(y_true) == len(y_predict), "y_true and y_predict must have same length"

    return np.sum(np.abs(y_true - y_predict)) / len(y_true)


def r2_score(y_true, y_predict):
    mse = mean_squared_error(y_true, y_predict)
    var = np.var(y_true)
    return 1 - mse / var


def TN(y_true, y_predict):
    assert len(y_true) == len(y_predict)
    return np.count_nonzero((y_true == 0) & (y_predict == 0))


def FP(y_true, y_predict):
    assert len(y_true) == len(y_predict)
    return np.count_nonzero((y_true == 0) & (y_predict == 1))


def FN(y_true, y_predict):
    assert len(y_true) == len(y_predict)
    return np.count_nonzero((y_true == 1) & (y_predict == 0))


def TP(y_true, y_predict):
    assert len(y_true) == len(y_predict)
    return np.count_nonzero((y_true == 1) & (y_predict == 1))


def confusion_matrix(y_true, y_predict):
    tp = TP(y_test, y_predict)
    fp = FP(y_test, y_predict)
    fn = FN(y_test, y_predict)
    tn = TN(y_test, y_predict)
    return np.array([[tn, fp],
                     [fn, tp]])


def precision_score(y_true, y_predict):
    tp = TP(y_true, y_predict)
    fp = FP(y_true, y_predict)
    try:
        return tp / (tp + fp)
    except:
        return 0.0


def recall_score(y_true, y_predict):
    tp = TP(y_true, y_predict)
    fn = FN(y_true, y_predict)
    try:
        return tp / (tp + fn)
    except:
        return 0.0


def f1_score(y_true, y_predict):
    precision = precision_score(y_true, y_predict)
    recall = recall_score(y_true, y_predict)

    try:
        return 2 * precision * recall / (precision + recall)
    except:
        return 0.0


def TPR(y_true, y_predict):
    tp = TP(y_true, y_predict)
    fn = FN(y_true, y_predict)
    try:
        return tp / (tp + fn)
    except:
        return 0.0


def FPR(y_true, y_predict):
    fp = FP(y_true, y_predict)
    tn = TN(y_true, y_predict)
    try:
        return fp / (fp + tn)
    except:
        return 0.0


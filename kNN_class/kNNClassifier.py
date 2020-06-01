import numpy as np
from math import sqrt
from collections import Counter


class KNNClassifier:

    def __init__(self, k):
        """初始化kNN分类器"""
        assert type(k) is int, "k must be type int"
        assert k > 0, "k must be greater than 0"
        self.k = k
        self._X_train = None
        self._y_train = None

    def fit(self, X_train, y_train):
        """根据训练数据集X_train和y_train训练kNN分类器"""
        assert (type(X_train) is np.ndarray) and (type(y_train) is np.ndarray), \
            "X_train and y_train must be type numpy.ndarray"
        assert (X_train.ndim == 2) and (y_train.ndim == 1), \
            "train_X and train_y must have dimensions of 2 and 1"
        assert 0 < self.k <= X_train.shape[0], "k must be > 0 and <= # of labeled examples"

        self._X_train = X_train
        self._y_train = y_train

        return self

    def predict(self, X_predict):
        """给定待预测数据集X_predict,返回X_predict的结果向量"""
        assert self._X_train is not None and self._y_train is not None, \
            "Model has not fit before predicting."
        assert type(X_predict) is np.ndarray, "X_predict must be type numpy.ndarray"
        assert X_predict.ndim == 2, "X_predict must have dimension of 2"
        assert X_predict.shape[1] == self._X_train.shape[1], \
            "Number of features in X_predict must match that in X_train"

        y_predict = [self._predict(x) for x in X_predict]
        return np.array(y_predict)

    def _predict(self, x):
        """给定单个待预测数据x,返回x的预测结果值"""
        distances = [sqrt(np.sum((x - x_train)**2)) for x_train in self._X_train]
        indices = np.argsort(distances)[:self.k]
        labels = [self._y_train[idx] for idx in indices]
        return Counter(labels).most_common(1)[0][0]

    def score(self, X_test, y_test):
        y_predict = self.predict(X_test)
        from ml_utils.metrics import accuracy_score
        return accuracy_score(y_test, y_predict)
    
    def __repr__(self):
        return "kNN(k=%d)" % self.k

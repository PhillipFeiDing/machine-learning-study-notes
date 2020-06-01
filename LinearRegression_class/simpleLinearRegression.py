import numpy as np


class SimpleLinearRegression:

    def __init__(self):
        """初始化Simple Linear Regression 模型"""
        self.a_ = None
        self.b_ = None

    def fit(self, x_train, y_train):
        """根据训练数据集x_train, y_train训练Simple Linear Regression模型"""
        assert type(x_train) is np.ndarray and type(y_train) is np.ndarray, \
            "x_train and y_train must be of type numpy.ndarray"
        assert x_train.ndim == 1 and y_train.ndim == 1 and len(x_train) == len(y_train), \
            "x_train and y_train must have dimensionality of 1, and they must be of same length"

        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)
        x_std = np.std(x_train)
        y_std = np.std(y_train)

        self.a_ = np.sum((x_train - x_mean) * (y_train - y_mean)) / np.sum((x_train - x_mean)**2)
        self.b_ = y_mean - self.a_ * x_mean

        return self

    def predict(self, x_predict):
        """给定待预测的数据集x_predict, 返回表示x_predict的结果的向量"""
        assert type(x_predict) is np.ndarray and x_predict.ndim == 1, \
            "x_predict must be a numpy.ndarray of dimensionality 1"
        assert self.a_ is not None and self.b_ is not None, "must fit before predicting"
        return x_predict * self.a_ + self.b_

    def _predict(self, x_single):
        return self.a_ * x_single + self.b_

    def __repr__(self):
        return "SimpleLinearRegression(a=%.5f, b=%.5f)" % (self.a_, self.b_)

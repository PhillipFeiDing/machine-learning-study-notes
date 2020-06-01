import numpy as np


class LinearRegression:

    def __init__(self):
        self.coeff_ = None
        self.intercept_ = None
        self._theta = None

    def fit_normal(self, X_train, y_train):
        """根据训练数据集X_train 和 y_train 训练Linear Regression模型"""
        assert type(X_train) is np.ndarray and type(y_train) is np.ndarray, \
            "X_train and y_train must be of type numpy.ndarray"
        assert X_train.ndim == 2 and y_train.ndim == 1, \
            "X_train and y_train must have dimensionality of 2 and 1, respectively."
        assert X_train.shape[0] == y_train.shape[0], "X_train must have same number of examples as y_train"

        X_b = np.hstack([np.full(shape=X_train.shape[0], fill_value=1).reshape(-1, 1), X_train])
        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)
        self.intercept_ = self._theta[0]
        self.coeff_ = self._theta[1:]

        return self
    
    def fit_gd(self, X_train, y_train, eta=0.01, n_iters=1e4):
        assert type(X_train) is np.ndarray and type(y_train) is np.ndarray, \
            "X_train and y_train must be of type numpy.ndarray"
        assert X_train.ndim == 2 and y_train.ndim == 1 and X_train.shape[0] == y_train.shape[0], \
            "X_train and y train must have dimensionality of 2 and 1, and they must have same number of examples"

        def J(theta, X_b, y):
            try:
                m = X_b.shape[0]
                return np.sum((X_b.dot(theta) - y)**2) / m
            except:
                return float('inf')

        def dJ(theta, X_b, y):
            diff = X_b.dot(theta) - y
            temp = np.array([X_b[:, col].dot(diff) for col in range(0, X_b.shape[1])])
            return temp * 2 / X_b.shape[0]

        def gradient_descent(X_b, y, initial_theta, eta, n_iters, epsilon=1e-8):
            theta = initial_theta
            last_theta = None
            iter_count = 0

            while (last_theta is None or abs(J(theta, X_b, y) - J(last_theta, X_b, y)) >= epsilon) and \
                iter_count < n_iters:
                print("\r%.2f%%" % (iter_count / n_iters * 100), end="")
                last_theta = theta
                gradient = dJ(theta, X_b, y)
                theta = theta - gradient * eta
                iter_count += 1

            print("\nDone")

            return theta

        X_b = np.hstack([np.ones(shape=(X_train.shape[0], 1)), X_train])
        initial_theta = np.zeros(X_b.shape[1])
        self._theta = gradient_descent(X_b, y_train, initial_theta, eta, n_iters)
        self.intercept_ = self._theta[0]
        self.coeff_ = self._theta[1:]

        return self

    def predict(self, X_predict):
        """给定待预测的数据集X_predict,返回表示X_predict的结果向量"""
        assert self.coeff_ is not None and self.intercept_ is not None and self._theta is not None, \
            "must fit before predicting"
        assert type(X_predict) is np.ndarray and X_predict.ndim == 2 and X_predict.shape[1] == len(self.coeff_), \
            "X_predict must be a numpy.ndarray with dimensionality of 2, and it must contain the same number of" + \
            " features as X_train"

        X_b = np.hstack([np.ones(X_predict.shape[0]).reshape(-1, 1), X_predict])
        return X_b.dot(self._theta)

    def score(self, X_test, y_test):
        """根据测试数据集 X_test 和 y_test 确定当前模型的准确度"""
        y_predict = self.predict(X_test)
        from sklearn.metrics import r2_score
        return r2_score(y_test, y_predict)

    def __repr__(self):
        return "LinearRegression(intercept=%.3f, coeff=%s)" % (self.intercept_, str(np.round(self.coeff_, 3)))


if __name__ == "__main__":
    from sklearn import datasets
    boston = datasets.load_boston()
    feature_col = list(boston.feature_names).index("RM")
    X = boston.data
    y = boston.target
    X = X[y < 50.0, :]
    y = y[y < 50.0]

    from sklearn.model_selection import train_test_split
    test_ratio = 0.2
    rand_seed = 666
    (X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=test_ratio, random_state=rand_seed)

    lin_reg = LinearRegression()
    lin_reg.fit_normal(X_train, y_train)
    print(lin_reg)

    print(lin_reg.score(X_test, y_test))

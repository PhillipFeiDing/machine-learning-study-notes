import numpy as np


class PCA:

    def __init__(self, n_components):
        """初始化PCA"""
        assert n_components > 0, "n_components must > 0"
        self.n_components = n_components
        self.components_ = None

    def fit(self, X, eta=0.01, n_iters=1e4):
        """获取数据集X的前n个主成分"""
        assert type(X) is np.ndarray and X.ndim == 2, \
            "X must be of type numpy.ndarray, and it must have dimensionality of 2"
        assert X.shape[1] >= self.n_components, "X must not have fewer features than n_components"

        def demean(X):
            return X - np.mean(X, axis=0)

        def direction(X):
            return X / np.linalg.norm(X)

        def f(w, X):
            return np.sum(X.dot(w) ** 2) / X.shape[0]

        def df(w, X):
            return X.T.dot(X.dot(w)) * 2 / X.shape[0]

        def first_component(X, initial_w, eta, n_iters=1e4, epsilon=1e-8):
            w = direction(initial_w)
            iter_count = 0

            while iter_count < n_iters:
                last_w = w
                gradient = df(w, X)
                w = direction(w + gradient * eta)
                if abs(f(w, X) - f(last_w, X)) < epsilon:
                    break
                iter_count = iter_count + 1

            return w

        X_pca = demean(X)
        self.components_ = np.empty(shape=(self.n_components, X.shape[1]))

        for i in range(0, self.n_components):
            initial_w = np.zeros(shape=X.shape[1], dtype=float)
            initial_w[0] = 1
            w = first_component(X_pca, initial_w, eta=eta)
            self.components_[i] = w
            X_pca = X_pca - X_pca.dot(w).reshape(-1, 1) * w

        return self

    def transform(self, X):
        """将给定的X，映射到各个主成分分量中"""
        assert X.shape[1] == self.components_.shape[1]

        return X.dot(self.components_.T)

    def inverse_transform(self, X):
        """将给定的X，反向映射回原来的特征空间"""
        assert X.shape[1] == self.components_.shape[0]

        return X.dot(self.components_)

    def __repr__(self):
        return "PCA(n_components=%d, components=%s)" % (self.n_components, np.round(self.components_, 3))
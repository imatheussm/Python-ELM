#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from elm import ELM


class MLELM(ELM):
    """
    Multi Layer Extreme Learning Machine
    """
    def __init__(self, *, hidden_neurons=None, a=1, random_state=None):
        super().__init__(hidden_neurons=hidden_neurons, a=a, random_state=random_state)

        self.betas = []
        self.elm = None
        self.out_num = None

    def __calc_hidden_layer(self, X):
        """
        Args:
        X np.array input feature vector
        """
        for beta in self.betas:
            X = np.dot(beta, X.T).T

        return X

    def fit(self, X, y):
        if self.hidden_neurons is None:
            self.hidden_neurons = 2 * X.shape[1]

        self.out_num = max(y)
        X = self._add_bias(X)

        for hid_num in self.hidden_neurons[:-1]:
            _X = self.__calc_hidden_layer(X)
            W = self._random_state.uniform(-1., 1., (hid_num, _X.shape[1]))
            H = np.linalg.pinv(self._sigmoid(np.dot(W, _X.T)))
            beta = np.dot(H.T, _X)
            self.betas.append(beta)

        _X = self.__calc_hidden_layer(X)

        self.elm = ELM(hidden_neurons=self.hidden_neurons[-1])
        self.elm.fit(_X, y)

        return self

    def predict(self, X):
        X = self.__calc_hidden_layer(self._add_bias(X))
        return self.elm.predict(X)


def main():
    from sklearn import preprocessing
    from sklearn.datasets import fetch_openml as fetch_mldata
    from sklearn.model_selection import train_test_split

    db_name = 'diabetes'
    data_set = fetch_mldata(db_name)
    data_set.data = preprocessing.normalize(data_set.data)

    tmp = data_set.target
    tmpL = [1 if i == "tested_positive" else -1 for i in tmp]
    data_set.target = tmpL

    X_train, X_test, y_train, y_test = train_test_split(data_set.data, data_set.target, test_size=0.4, random_state=0)

    mlelm = MLELM(hidden_neurons=(10, 30, 200), random_state=0).fit(X_train, y_train)
    elm = ELM(hidden_neurons=200, random_state=0).fit(X_train, y_train)

    print("MLELM Accuracy %0.3f " % mlelm.score(X_test, y_test))
    print("ELM Accuracy %0.3f " % elm.score(X_test, y_test))


if __name__ == "__main__":
    main()

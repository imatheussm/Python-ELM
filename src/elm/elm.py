#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Extreme Learning Machine
This script is ELM for binary and multiclass classification.
"""

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_random_state


class ELM(BaseEstimator, ClassifierMixin):
    """
    3 step model ELM
    """
    def __init__(self, *, hidden_neurons=None, a=1, random_state=None):
        """
        Args:
        hid_num (int): number of hidden neurons
        a (int) : const value of sigmoid function
        """
        self.hidden_neurons = hidden_neurons
        self.a = a
        self.random_state = random_state

        self.beta = None
        self.out_num = None
        self.W = None

    @property
    def random_state(self):
        return self.__random_state_value

    @random_state.setter
    def random_state(self, new_random_state_value):
        if new_random_state_value is not None and type(new_random_state_value) is not int:
            raise TypeError("The random_state attribute must receive an integer object.")

        self.__random_state_value = new_random_state_value
        self.__random_state = check_random_state(new_random_state_value)

    @property
    def _random_state(self):
        return self.__random_state

    def _sigmoid(self, x):
        """
        sigmoid function
        Args:
        x (float): input

        Returns:
        float: output of sigmoid
        """
        sigmoid_range = 34.538776394910684
        x = np.clip(x, -sigmoid_range, sigmoid_range)

        return 1 / (1 + np.exp(-self.a * x))

    @classmethod
    def _add_bias(cls, X):
        """add bias to list

        Args:
        x_vs [[float]] Array: vec to add bias

        Returns:
        [float]: added vec

        Examples:
        >>> ELM._add_bias(np.array([[1,2,3], [1,2,3]]))
        array([[1., 2., 3., 1.],
               [1., 2., 3., 1.]])
        """

        return np.c_[X, np.ones(X.shape[0])]

    @classmethod
    def _label_scalar_to_vector(cls, n, label):
        """
        trasform label scalar to vector
        Args:
        n (int) : number of class, number of out layer neuron
        label (int) : label

        Exmples:
        >>> ELM._label_scalar_to_vector(3, 1)
        [1, -1, -1]
        >>> ELM._label_scalar_to_vector(3, 2)
        [-1, 1, -1]
        >>> ELM._label_scalar_to_vector(3, 3)
        [-1, -1, 1]
        """
        return [-1 if i != label else 1 for i in range(1, n + 1)]

    def fit(self, X, y):
        """
        learning

        Args:
        X [[float]] array : feature vectors of learning data
        y [[float]] array : labels of leaning data
        """
        if self.hidden_neurons is None:
            self.hidden_neurons = 2 * X.shape[1]

        # number of class, number of output neuron
        self.out_num = max(y)

        if self.out_num != 1:
            y = np.array([self._label_scalar_to_vector(self.out_num, _y) for _y in y])

        # add bias to feature vectors
        X = self._add_bias(X)

        # generate weights between input layer and hidden layer
        self.W = self._random_state.uniform(-1., 1., (self.hidden_neurons, X.shape[1]))

        # find inverse weight matrix
        H = np.linalg.pinv(self._sigmoid(np.dot(self.W, X.T)))
        self.beta = np.dot(H.T, y)

        return self

    def predict(self, X):
        """
        predict classify result

        Args:
        X [[float]] array: feature vectors of learning data

        Returns:
        [int]: labels of classification result
        """
        H = self._sigmoid(np.dot(self.W, self._add_bias(X).T))
        y = np.dot(H.T, self.beta)

        if self.out_num == 1:
            return np.sign(y)
        else:
            return np.argmax(y, 1) + np.ones(y.shape[0])


def main():
    from sklearn import preprocessing
    from sklearn.datasets import fetch_openml as fetch_mldata
    from sklearn.model_selection import KFold, cross_val_score

    db_name = 'australian'
    hid_nums = [None, 100, 200, 300]

    data_set = fetch_mldata(db_name)
    data_set.data = preprocessing.normalize(data_set.data)
    data_set.target = [1 if i == 1 else -1 for i in data_set.target.astype(int)]

    for hid_num in hid_nums:
        print(hid_num, end=' ')

        e = ELM(hidden_neurons=hid_num)

        ave = 0
        for i in range(10):
            cv = KFold(n_splits=5, shuffle=True)
            scores = cross_val_score(e, data_set.data, data_set.target, cv=cv, scoring='accuracy', n_jobs=-1)
            ave += scores.mean()

        ave /= 10

        print("Accuracy: %0.3f " % ave)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
    main()

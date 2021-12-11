import numpy as np


class KNN:

    def __init__(self, k=5):
        """
        Initializes KNN class
        :param k: number of nearest neighbors
        """
        self.k = k

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fits the model to the training data
        :param X: training data
        :param y: training labels
        """
        self.X_train: np.ndarray = X
        self.y_train: np.ndarray = y

    def predict(self, X: np.ndarray):
        """
        Predicts the class labels for the provided data
        :param X: data to predict on
        :return: class labels for the data
        """
        dists = self.compute_distances(X)
        return self.predict_labels(dists)

    def compute_distances(self, X):
        """
        Computes the distance between test point X and each training point in self.X_train
        :param X: test data
        :return: array of shape (num_train,) distance between test point and the jth training point.
        """
        return np.sum((X - self.X_train) ** 2, axis=1)
        
    def predict_labels(self, dists: np.ndarray):
        """
        Given a matrix of distances between the test and training points, predict a label for the test point.
        :param dists: distance matrix
        :return: predicted labels for the data in dists
        """
        order = np.argsort(dists)
        closest_y = self.y_train[order[:self.k]]
        return np.argmax(np.bincount(closest_y))

    def predict_batch(self, batch):
        """
        Given a list of test points, predicts a label for each one.
        :param batch: list of test points
        :return: list of predicted labels
        """
        return np.array([self.predict(x) for x in batch])
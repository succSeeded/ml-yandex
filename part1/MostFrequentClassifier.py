import numpy as np
from sklearn.base import ClassifierMixin


class MostFrequentClassifier(ClassifierMixin):
    # Predicts the rounded (just in case) median of y_train
    def fit(self, X=None, y=None):
        """
        Parameters
        ----------
        X : array like, shape = (n_samples, n_features)
        Training data features
        y : array like, shape = (_samples,)
        Training data targets
        """
        uniques, counts = np.unique(y, return_counts=True)
        self.mode_ = uniques[np.argmax(counts)]

    def predict(self, X=None):
        """
        Parameters
        ----------
        X : array like, shape = (n_samples, n_features)
        Data to predict
        """
        return np.ones((X.shape[0],)) * self.mode_
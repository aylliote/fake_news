from sklearn.linear_model import LogisticRegressionCV
import numpy as np
from sklearn.base import BaseEstimator


def new_score(p):
    if p[1] > 0.5:
        return [0, 0, 0, 0, 1, 0]
    else:
        return [0, 1, 0, 0, 0, 0]


class Classifier(BaseEstimator):
    def __init__(self):
        self.clf = LogisticRegressionCV(Cs=[0.04] * 4, cv=4)

    def fit(self, X, y):
        self.clf.fit(X, (y > 2).astype(int))

    def predict(self, X):
        y_pred = self.clf.predict_proba(X)
        return np.array([np.argmax(new_score(p)) for p in y_pred])

    def predict_proba(self, X):
        y_pred = self.clf.predict_proba(X)
        return np.array([new_score(p) for p in y_pred])

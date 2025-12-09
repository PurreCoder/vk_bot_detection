import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class MinMaxTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, Xmin, Xmax):
        self.Xmin = Xmin
        self.Xmax = Xmax

    def fit(self, X):
        return self

    def transform(self, X):
        res = (X - self.Xmin) / (self.Xmax - self.Xmin)
        np.clip(res, 0, 1, res)
        return res


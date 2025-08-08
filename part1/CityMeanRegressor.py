import numpy as np
from sklearn.base import RegressorMixin


class CityMeanRegressor(RegressorMixin):
    def fit(self, X=None, y=None):
        msk_mask = X["city"] == "msk"
        self.msk_mean_ = y[msk_mask].mean()
        self.spb_mean_ = y[~msk_mask].mean()

    def predict(self, X=None):
        msk_mask = X["city"] == "msk"
        ans = np.ones((X.shape[0],))
        ans[msk_mask] = self.msk_mean_
        ans[~msk_mask] = self.spb_mean_
        return ans
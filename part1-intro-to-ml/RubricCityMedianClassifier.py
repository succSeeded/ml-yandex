import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin


class RubricCityMedianClassifier(ClassifierMixin):
    def fit(self, X=None, y=None):
        msk_mask = X["city"] == "msk"
        self.mean_dict_ = (
            X[["city", "modified_rubrics"]]
            .join(y, how="left")
            .groupby(by=["city", "modified_rubrics"])
        ).median()

    def predict(self, X=None):
        return pd.merge(
            X.drop(axis=1, columns=["average_bill"]),
            self.mean_dict_,
            how="left",
            left_on=["city", "modified_rubrics"],
            right_index=True,
        )["average_bill"]
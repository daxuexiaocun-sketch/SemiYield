"""Leakage-safe preprocessing components."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class ColumnCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, max_missing: float = 0.5, max_features: int | None = None):
        self.max_missing = max_missing
        self.max_features = max_features

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None):
        frame = pd.DataFrame(X)
        keep = frame.columns[
            (frame.isna().mean() <= self.max_missing) & (frame.nunique(dropna=True) > 1)
        ]
        if self.max_features and len(keep) > self.max_features:
            if y is None:
                keep = keep[: self.max_features]
            else:
                from sklearn.feature_selection import mutual_info_classif

                imputed = SimpleImputer(strategy="median").fit_transform(frame.loc[:, keep])
                scores = mutual_info_classif(imputed, y, random_state=42)
                keep = keep[np.argsort(scores)[::-1][: self.max_features]]
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        self.selected_columns_ = np.asarray(keep, dtype=object)
        if not len(self.selected_columns_):
            raise ValueError("No usable features remain after cleaning")
        return self

    def transform(self, X: pd.DataFrame):
        frame = pd.DataFrame(X, columns=getattr(X, "columns", self.feature_names_in_))
        return frame.loc[:, self.selected_columns_]

    def get_feature_names_out(self, input_features=None):
        return self.selected_columns_


def build_preprocessor(*, scale: bool = True, max_features: int | None = None) -> Pipeline:
    steps = [
        ("clean", ColumnCleaner(max_missing=0.5, max_features=max_features)),
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
    ]
    if scale:
        steps.append(("scale", StandardScaler()))
    return Pipeline(steps)

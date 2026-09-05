"""Leakage-safe preprocessing for mixed packaging process inputs."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from semiyield.packaging.data import CATEGORICAL, NUMERIC


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="__missing__")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [("numeric", numeric, NUMERIC), ("categorical", categorical, CATEGORICAL)]
    )

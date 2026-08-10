import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
 
from processing.features import build_preprocessor, ID_COLUMNS
 
 
def build_pipeline(x: pd.DataFrame, config: dict = None) -> Pipeline:
    """
    Build the full preprocessing + model pipeline.
 
    Drops ID_COLUMNS before fitting (they're identifiers, not features,
    and LogisticRegression can't handle their raw string dtype), one-hot
    encodes the remaining categorical columns, and feeds them into a
    tuned LogisticRegression (C=0.1, per the notebook's RandomizedSearchCV
    result on this dataset's small sample size).
    """
    model_params = (config or {}).get("model", {}).get("params", {
        "C": 0.1,
        "penalty": "l2",
        "solver": "saga",
        "max_iter": 1000,
        "random_state": 42,
    })
 
    x_features = x.drop(columns=[c for c in ID_COLUMNS if c in x.columns])
    preprocessor = build_preprocessor(x_features)
 
    return Pipeline([
        ("transformer", preprocessor),
        ("model", LogisticRegression(**model_params)),
    ])
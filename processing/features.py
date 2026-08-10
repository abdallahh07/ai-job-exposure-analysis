
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
 
# Columns that are identifiers, not features — never one-hot encode these.
ID_COLUMNS = ["occupation_title", "soc_code"]
 
 
def get_categorical_columns(x: pd.DataFrame, exclude: list[str] = None) -> list[str]:
    """Return object-dtype columns in x, excluding identifier columns."""
    exclude = exclude or ID_COLUMNS
    obj_cols = x.select_dtypes(include="object").columns.tolist()
    return [col for col in obj_cols if col not in exclude]
 
 
def build_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    """Build the ColumnTransformer that one-hot encodes categorical columns."""
    categorical_cols = get_categorical_columns(x)
 
    one_hot = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
 
    return ColumnTransformer([
        ("one_hot", one_hot, categorical_cols),
    ])
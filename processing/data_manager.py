
import os
import yaml
import joblib
import pandas as pd
 
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "config.yml")
 
 
def load_config(config_path: str = CONFIG_PATH) -> dict:
    """Load project configuration (data paths, label mapping, model path) from YAML."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
 
 
def load_raw(config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the three raw source CSVs as-is."""
    data_dir = config["data"]["data_dir"]
 
    ai_exposure = pd.read_csv(os.path.join(data_dir, config["data"]["ai_exposure_file"]))
    cognitive_ability = pd.read_csv(os.path.join(data_dir, config["data"]["cognitive_ability_file"]))
    occupation_cognitive = pd.read_csv(os.path.join(data_dir, config["data"]["occupation_cognitive_file"]))
 
    return ai_exposure, cognitive_ability, occupation_cognitive
 
 
def merge_datasets(ai_exposure: pd.DataFrame, occupation_cognitive: pd.DataFrame) -> pd.DataFrame:
    """Merge ai_exposure with occupation_cognitive on soc_code + occupation_title."""
    return pd.merge(
        ai_exposure,
        occupation_cognitive,
        on=["soc_code", "occupation_title"],
        how="left",
    )
 
 
def load_dataset(config: dict = None) -> pd.DataFrame:
    """Convenience wrapper: load raw CSVs and return the merged dataset ready for splitting."""
    config = config or load_config()
    ai_exposure, _, occupation_cognitive = load_raw(config)
    return merge_datasets(ai_exposure, occupation_cognitive)
 
 
def get_x_y(df: pd.DataFrame, config: dict = None) -> tuple[pd.DataFrame, pd.Series]:
    """Split the merged dataset into X (features) and y (encoded ai_exposure_level)."""
    config = config or load_config()
 
    leak_cols = config["features"]["drop_columns"]
    label_map = config["features"]["label_map"]
 
    x = df.drop(columns=leak_cols)
    y = df[config["features"]["target_column"]].map(label_map)
 
    return x, y
 
 
def save_pipeline(pipeline, config: dict = None) -> str:
    """Persist a fitted pipeline/model to disk via joblib. Returns the output path."""
    config = config or load_config()
    output_path = config["model"]["output_path"]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(pipeline, output_path)
    return output_path
 
 
def load_pipeline(config: dict = None):
    """Load a previously saved pipeline/model from disk via joblib."""
    config = config or load_config()
    return joblib.load(config["model"]["output_path"])
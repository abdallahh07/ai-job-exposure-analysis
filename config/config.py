from pathlib import Path
from typing import List
from pydantic import BaseModel
import yaml

PACKAGE_ROOT = Path(__file__).parent
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yml"

class AppConfig(BaseModel):
    pipeline_name: str
    pipeline_save_file: str
    data_folder: str
    ai_exposure: str
    cognitive_ability: str
    occupation_cognitive: str
    target: str
    test_size: float
    features_to_drop: List[str]
    # categorical_features: List[str]  # add once you pull the real list from your notebook

def fetch_config_from_yaml() -> AppConfig:
    with open(CONFIG_FILE_PATH, "r") as f:
        parsed = yaml.safe_load(f)
        return AppConfig(**parsed)

config = fetch_config_from_yaml()
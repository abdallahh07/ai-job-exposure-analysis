import pandas as pd
 
from processing.data_manager import load_config, load_pipeline
 
# Inverse of the label_map in config.yml (0/1/2 -> Low/Medium/High)
LEVEL_LABELS = {0: "Low", 1: "Medium", 2: "High"}
 
 
def predict(input_data: pd.DataFrame, config: dict = None) -> pd.DataFrame:
    """
    Run predictions on new occupation records.
 
    input_data must have the same feature columns the pipeline was
    trained on (everything except the dropped leakage columns and
    ai_exposure_level itself — ID_COLUMNS are dropped internally by
    the pipeline's transformer step, so they can be left in or out).
 
    Returns input_data with two added columns:
      - ai_exposure_level_pred: predicted label (Low/Medium/High)
      - ai_exposure_level_pred_code: raw predicted class (0/1/2)
    """
    config = config or load_config()
    pipe = load_pipeline(config)
 
    preds = pipe.predict(input_data)
 
    result = input_data.copy()
    result["ai_exposure_level_pred_code"] = preds
    result["ai_exposure_level_pred"] = result["ai_exposure_level_pred_code"].map(LEVEL_LABELS)
 
    return result
 
 
if __name__ == "__main__":
    import sys
 
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_csv>")
        sys.exit(1)
 
    df = pd.read_csv(sys.argv[1])
    output = predict(df)
    print(output[["occupation_title", "ai_exposure_level_pred"]])
 


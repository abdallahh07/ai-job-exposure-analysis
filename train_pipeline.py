from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
 
from processing.data_manager import load_config, load_dataset, get_x_y, save_pipeline
from pipeline import build_pipeline
 
 
def run_training():
    config = load_config()
 
    df = load_dataset(config)
    x, y = get_x_y(df, config)
 
    x_train, x_test, y_train, y_test = train_test_split(
        x, y,
        test_size=config.get("training", {}).get("test_size", 0.2),
        random_state=config.get("training", {}).get("random_state", 42),
        stratify=y,
    )
 
    pipe = build_pipeline(x_train, config)
    pipe.fit(x_train, y_train)
 
    y_pred = pipe.predict(x_test)
    train_score = pipe.score(x_train, y_train)
    test_acc = accuracy_score(y_test, y_pred)
 
    print(f"Train accuracy: {train_score:.4f}")
    print(f"Test accuracy:  {test_acc:.4f}")
    print(classification_report(y_test, y_pred))
 
    output_path = save_pipeline(pipe, config)
    print(f"Saved trained pipeline to {output_path}")
 
    return pipe
 
 
if __name__ == "__main__":
    run_training()
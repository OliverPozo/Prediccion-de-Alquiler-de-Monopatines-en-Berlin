import pandas as pd 
import numpy as np
import joblib
import os
from src.models.model_factory import ModelFactory
from src.models.evaluator import ModelEvaluator
from src.data_cleaning import cleaning
from config.hyperparameters import MODEL_PARAMS
from config.data_config import CONFIG_DATA


def load_and_preprocess_data(data_path):
    df = pd.read_csv(data_path)
    df = cleaning(df)
    df.to_csv("data/processed/dataset_alquiler_cleaned.csv", index=False)
    return df

def prepare_features(df, target_column, log_transform=False):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    y.to_csv("data/processed/y_real.csv", index=False)
    if log_transform:
        y = np.log1p(y)
    return X, y

def train_models(X_train, y_train, X_val, y_val, preprocessor, models_config, model_params, log_transformed=False):
    results = {}
    for model_name in models_config:
        print(f"\nTraining {model_name}...")
        model = ModelFactory.create_model(model_name, preprocessor, model_params.get(model_name, {}))
        pipeline = model.build_pipeline()
        pipeline.fit(X_train, y_train)
        
        metrics = ModelEvaluator.evaluate_train_validate(pipeline, X_train, y_train, X_val, y_val, log_transformed)
        results[model_name] = {
            "pipeline": pipeline,
            "metrics": metrics
        }
        print(metrics)
    return results

def evaluate_final_model(model, X_test, y_test, log_transformed=False):
    return ModelEvaluator.evaluate(model, X_test, y_test, "test", log_transformed)

def save_model(model, metadata, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump({"pipeline": model, "metadata": metadata}, path)


def save_all_models(results):
    for model_name, result in results.items():
        save_model(
            result["pipeline"],
            {"metrics": result["metrics"]},
            f"saved_models/{model_name}.pkl"
        )
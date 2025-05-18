from src.model_pipeline_utils import load_and_preprocess_data, prepare_features, train_models, evaluate_final_model, save_model, save_all_models
from sklearn.model_selection import train_test_split
from src.feature_engineering import create_preprocessor
from config.data_config import CONFIG_DATA 
from config.hyperparameters import MODEL_PARAMS

def run_pipeline():
    # 1. Carga y preprocesamiento
    df = load_and_preprocess_data(CONFIG_DATA['data_path'])
    
    # 2. Preparación de features
    X, y = prepare_features(df, CONFIG_DATA['target_column'], CONFIG_DATA['log_transform'])
    
    # 3. División de datos
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=CONFIG_DATA['test_size'], random_state=CONFIG_DATA['random_state'])
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=CONFIG_DATA['val_size'], random_state=CONFIG_DATA['random_state'])

    X_test.to_csv("data/processed/X_test.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)
    X_val.to_csv("data/processed/X_val.csv", index=False)
    y_val.to_csv("data/processed/y_val.csv", index=False)

    # 4. Preprocesamiento
    preprocessor = create_preprocessor(
        CONFIG_DATA['numeric_features'],
        CONFIG_DATA['categorical_features'],
        scaler=CONFIG_DATA['scaler']
    )
    
    # 5. Entrenamiento
    results = train_models(X_train, y_train, X_val, y_val, preprocessor, CONFIG_DATA['models'], MODEL_PARAMS, CONFIG_DATA['log_transform'])

    # 6. Evaluación final
    best_model = min(results, key=lambda k: results[k]["metrics"].loc["validation", "RMSE"])
    print("\n" + "-"*50 + "\n")
    print(f"Best model: {best_model}")
    print(f"Validation RMSE: {results[best_model]['metrics'].loc['validation', 'RMSE']}")
    test_metrics = evaluate_final_model(results[best_model]["pipeline"], X_test, y_test, CONFIG_DATA['log_transform'])
   
    # 7. Guardar todos los modelos
    save_all_models(results)
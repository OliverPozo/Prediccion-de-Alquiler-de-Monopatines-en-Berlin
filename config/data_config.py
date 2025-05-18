
CONFIG_DATA = {
    'data_path': 'data/original/dataset_alquiler.csv',
    'target_column': 'total_alquileres',
    "numeric_features": ['temperatura', 'humedad', 'velocidad_viento'],
    'categorical_features': ['temporada', 'anio', 'mes','hora','feriado','dia_semana','dia_trabajo', 'clima'], 
    'test_size': 0.3,
    'val_size': 0.5,
    'random_state': 42,
    'log_transform': True, 
    'scaler': {
        'type': 'StandardScaler',
        'params': {}
    },
    'search_type': 'grid',
    #'models': ['lasso', 'elasticnet', 'ridge', 'gradient_boosting', 'random_forest'],
    'models': ['linear', 'lasso', 'ridge', 'elasticnet', 'gradient_boosting', 'random_forest', 'gradient_boosting_hybrid'],
    #'models': ['linear', 'lasso', 'elasticnet', 'ridge']
    #Trees
    #'models':['random_forest', 'gradient_boosting', 'gradient_boosting_hybrid'],
    #'models': ['linear']
}
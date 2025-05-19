from scipy.stats import randint, uniform

MODEL_PARAMS = {
    'lasso': {
        'alpha': [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0],
        'cv': 5
    },
    'elasticnet': {
        'alpha': [1e-4, 1e-3, 1e-2, 0.1, 1.0],
        'l1_ratio': [0.1, 0.25, 0.5, 0.75, 0.9],
        'cv': 5
    },
    'ridge': {
        'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
        'cv': 5
    },
    'gradient_boosting': {
        'n_estimators': randint(50, 301),           
        'learning_rate': uniform(0.01, 0.19),       
        'max_depth': randint(3, 11),                 
        'subsample': uniform(0.6, 0.4),              
        'cv': 5
    },
    'random_forest': {
        'n_estimators': randint(50, 301),            
        'max_depth': [10, 20, None],                 
        'min_samples_split': randint(2, 11),         
        'min_samples_leaf': randint(1, 5),           
        'cv': 5
    },
    'gradient_boosting_hybrid': {
        'n_estimators': randint(50, 301),            
        'learning_rate': uniform(0.01, 0.19),        
        'max_depth': randint(3, 11),                 
        'subsample': uniform(0.6, 0.4),              
        'cv': 5
    }
}

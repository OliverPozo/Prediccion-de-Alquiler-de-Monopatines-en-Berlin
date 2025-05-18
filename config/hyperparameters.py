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
        'n_estimators': randint(50, 301),           # entre 50 y 300
        'learning_rate': uniform(0.01, 0.19),        # entre 0.01 y 0.2
        'max_depth': randint(3, 11),                 # entre 3 y 10
        'subsample': uniform(0.6, 0.4),              # entre 0.6 y 1.0
        'cv': 5
    },
    'random_forest': {
        'n_estimators': randint(50, 301),            # entre 50 y 200
        'max_depth': [10, 20, None],                 # fija, puede mantenerse así
        'min_samples_split': randint(2, 11),         # entre 2 y 10
        'min_samples_leaf': randint(1, 5),           # entre 1 y 4
        'cv': 5
    },
    'gradient_boosting_hybrid': {
        'n_estimators': randint(50, 301),            # entre 50 y 100
        'learning_rate': uniform(0.01, 0.19),        # entre 0.01 y 0.1
        'max_depth': randint(3, 11),                  # entre 3 y 7
        'subsample': uniform(0.6, 0.4),              # entre 0.6 y 1.0
        'cv': 5
    }
}

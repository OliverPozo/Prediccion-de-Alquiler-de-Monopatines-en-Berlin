from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from .base_model import BaseModel
import numpy as np

class GradientBoostingModel(BaseModel):
    def __init__(self, preprocessor, **params):
        super().__init__(preprocessor)
        self.params = params
        
    def build_model(self):
        return RandomizedSearchCV(
            GradientBoostingRegressor(),
            param_distributions={
                'n_estimators': self.params['n_estimators'],
                'learning_rate': self.params['learning_rate'],
                'max_depth': self.params['max_depth']
            },
            n_iter=self.params.get('n_iter', 10),
            cv=self.params.get('cv', 3),
            scoring='neg_root_mean_squared_error',
            n_jobs=-1,
            random_state=self.params.get('random_state', 42)
        )

class RandomForestModel(BaseModel):
    def __init__(self, preprocessor, **params):
        super().__init__(preprocessor)
        self.params = params
        
    def build_model(self):
        return RandomizedSearchCV(
            RandomForestRegressor(),
            param_distributions={
                'n_estimators': self.params['n_estimators'],
                'max_depth': self.params['max_depth'],
                'min_samples_split': self.params['min_samples_split'],
                'min_samples_leaf': self.params['min_samples_leaf']
            },
            n_iter=self.params.get('n_iter', 10),
            cv=self.params.get('cv', 3),
            scoring='neg_root_mean_squared_error',
            n_jobs=-1,
            random_state=self.params.get('random_state', 42)
        )

class GradientBoostingHybridModel(BaseModel):
    def __init__(self, preprocessor, **params):
        super().__init__(preprocessor)
        self.params = params
        self.cv = params.get('cv', 3)
        
    def build_model(self):
        return RandomizedSearchCV(
            GradientBoostingRegressor(),
            param_distributions={
                'n_estimators': self.params['n_estimators'],
                'learning_rate': self.params['learning_rate'],
                'max_depth': self.params['max_depth'],
                'subsample': self.params.get('subsample', [0.6, 0.8, 1.0])
            },
            n_iter=self.params.get('n_iter', 20),
            cv=self.cv,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1,
            random_state=self.params.get('random_state', 42)
        )

    def fit(self, X, y):
        # Fase 1: RandomizedSearch
        self.pipeline = self.build_pipeline()
        self.pipeline.fit(X, y)

        # Fase 2: GridSearch refinado
        best_params = self.pipeline.named_steps['model'].best_params_
        
        param_grid = {
            'n_estimators': np.unique([
                max(10, best_params['n_estimators'] - 50),
                best_params['n_estimators'],
                best_params['n_estimators'] + 50
            ]).tolist(),
            'max_depth': np.unique([
                max(1, best_params['max_depth'] - 1),
                best_params['max_depth'],
                best_params['max_depth'] + 1
            ]).tolist(),
            'learning_rate': np.unique([
                round(best_params['learning_rate'] * 0.5, 3),
                best_params['learning_rate'],
                round(best_params['learning_rate'] * 1.5, 3)
            ]).tolist()
        }
        
        grid_search = GridSearchCV(
            GradientBoostingRegressor(
                subsample=best_params.get('subsample', 1.0)
            ),
            param_grid=param_grid,
            cv=self.cv,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1
        )
        
        self.pipeline.steps[-1] = ('model', grid_search)
        self.pipeline.fit(X, y)
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from .base_model import BaseModel

class LinearRegressionModel(BaseModel):
    def build_model(self):
        return LinearRegression()

class RidgeModel(BaseModel):
    def __init__(self, preprocessor, **params):
        super().__init__(preprocessor)
        self.params = params

    def build_model(self):
        search_type = self.params.get('search_type', 'grid')
        search_params = {'alpha': self.params['alpha']}
        cv = self.params.get('cv', 3)

        if search_type == 'random':
            return RandomizedSearchCV(
                Ridge(max_iter=10000),
                param_distributions=search_params,
                n_iter=self.params.get('n_iter', 20),
                cv=cv,
                scoring='neg_root_mean_squared_error',
                random_state=42,
                n_jobs=-1
            )
        else:
            return GridSearchCV(
                Ridge(max_iter=10000),
                param_grid=search_params,
                cv=cv,
                scoring='neg_root_mean_squared_error',
                n_jobs=-1
            )

class LassoModel(BaseModel):
    def __init__(self, preprocessor, **params):
        super().__init__(preprocessor)
        self.params = params

    def build_model(self):
        search_type = self.params.get('search_type', 'grid')
        search_params = {'alpha': self.params['alpha']}
        cv = self.params.get('cv', 3)

        if search_type == 'random':
            return RandomizedSearchCV(
                Lasso(max_iter=10000),
                param_distributions=search_params,
                n_iter=self.params.get('n_iter', 20),
                cv=cv,
                scoring='neg_root_mean_squared_error',
                random_state=42,
                n_jobs=-1
            )
        else:
            return GridSearchCV(
                Lasso(max_iter=10000),
                param_grid=search_params,
                cv=cv,
                scoring='neg_root_mean_squared_error',
                n_jobs=-1
            )

class ElasticNetModel(BaseModel):
    def __init__(self, preprocessor, **params):
        super().__init__(preprocessor)
        self.params = params

    def build_model(self):
        search_type = self.params.get('search_type', 'grid')
        search_params = {
            'alpha': self.params['alpha'],
            'l1_ratio': self.params['l1_ratio']
        }
        cv = self.params.get('cv', 3)

        if search_type == 'random':
            return RandomizedSearchCV(
                ElasticNet(max_iter=10000),
                param_distributions=search_params,
                n_iter=self.params.get('n_iter', 20),
                cv=cv,
                scoring='neg_root_mean_squared_error',
                random_state=42,
                n_jobs=-1
            )
        else:
            return GridSearchCV(
                ElasticNet(max_iter=10000),
                param_grid=search_params,
                cv=cv,
                scoring='neg_root_mean_squared_error',
                n_jobs=-1
            )

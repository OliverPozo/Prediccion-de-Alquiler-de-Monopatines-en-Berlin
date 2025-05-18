from .linear_models import LinearRegressionModel, RidgeModel, LassoModel, ElasticNetModel
from .ensemble_models import RandomForestModel, GradientBoostingModel

class ModelFactory:
    @staticmethod
    def create_model(model_name, preprocessor, model_params=None):
        """Crea una instancia del modelo con los parámetros especificados.
        
        Args:
            model_name (str): Nombre del modelo a crear
            preprocessor: Pipeline de preprocesamiento
            model_params (dict): Parámetros del modelo (opcional)
            
        Returns:
            Instancia del modelo configurado
        """
        if model_params is None:
            model_params = {}
            
        models = {
            'linear': LinearRegressionModel(preprocessor),
            'ridge': RidgeModel(preprocessor, **model_params),
            'lasso': LassoModel(preprocessor, **model_params),
            'elasticnet': ElasticNetModel(preprocessor, **model_params),
            'gradient_boosting': GradientBoostingModel(preprocessor, **model_params),
            'random_forest': RandomForestModel(preprocessor, **model_params),
            'gradient_boosting_hybrid': GradientBoostingModel(preprocessor, **model_params),
        }
        
        try:
            return models[model_name]
        except KeyError:
            raise ValueError(f"Modelo no soportado: {model_name}. Opciones disponibles: {list(models.keys())}")
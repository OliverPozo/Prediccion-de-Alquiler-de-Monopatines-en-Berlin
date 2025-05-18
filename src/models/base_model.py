from abc import ABC, abstractmethod
from sklearn.pipeline import Pipeline

class BaseModel(ABC):
    def __init__(self, preprocessor):
        self.pipeline = None
        self.preprocessor = preprocessor

    @abstractmethod
    def build_model(self):
        """Construye el modelo específico."""
        pass

    def build_pipeline(self):
        """Crea el pipeline sklearn."""
        self.pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('model', self.build_model())
        ])
        return self.pipeline

    def fit(self, X, y):
        self.pipeline.fit(X, y)
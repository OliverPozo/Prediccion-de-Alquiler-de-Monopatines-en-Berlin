import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

class ModelEvaluator:
    @staticmethod
    def evaluate(pipeline, X, y, split_name="test", log_transformed_target=False):
        """
        Evalúa un pipeline completo (preprocesador + modelo)
        
        Args:
            pipeline: Pipeline de sklearn con preprocesamiento y modelo
            X: Features
            y: Target (en escala log si log_transformed_target=True)
            log_transformed_target: Indica si 'y' está en escala logarítmica
        """
        preds = pipeline.predict(X)
        
        if log_transformed_target:
            # Destransformamos predicciones y valores reales
            preds = np.expm1(preds)
            y = np.expm1(y)
        
        metrics = {
            'MAE': mean_absolute_error(y, preds),
            'RMSE': np.sqrt(mean_squared_error(y, preds)),
            'R2': r2_score(y, preds)
        }
        if hasattr(pipeline.named_steps.get('model', None), 'best_params_'):
            metrics['best_params'] = pipeline.named_steps['model'].best_params_
        
        print(f"Evaluating {split_name} set:")
        # Para mostrar mejor el diccionario en best_params al imprimir en consola:
        for k, v in metrics.items():
            if k == 'best_params':
                print(f"{k}: {v}")  # imprime completo sin truncar
            else:
                print(f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}")
            
        return metrics

    @staticmethod
    def evaluate_train_validate(pipeline, X_train, y_train, X_val, y_val, log_transformed_target=False):
        """
        Evalúa el pipeline en train y validation, con opción para destransformar targets logarítmicos

        Args:
            log_transformed_target: Si True, aplica np.expm1() a targets y predicciones
        """
        results = {
            'train': ModelEvaluator.evaluate(pipeline, X_train, y_train, "train", log_transformed_target),
            'validation': ModelEvaluator.evaluate(pipeline, X_val, y_val, "validation", log_transformed_target)
        }
        df_results = pd.DataFrame(results).transpose()
        
        # Para evitar truncamiento en pandas al mostrar 'best_params'
        pd.set_option('display.max_colwidth', None)
        
        return df_results

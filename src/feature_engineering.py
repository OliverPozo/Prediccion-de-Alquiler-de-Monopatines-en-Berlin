# features.py
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def create_preprocessor(numeric_features, categorical_features, scaler):    
    """
    Create a preprocessing pipeline for the data.
    """
    numeric_transformer = StandardScaler(**scaler['params']) if scaler['type'] == 'StandardScaler' else None
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    return preprocessor

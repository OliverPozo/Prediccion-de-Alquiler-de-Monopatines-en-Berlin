from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.data_cleaning import cleaning
import pandas as pd
import numpy as np
import joblib


MODELOS = {
    "linear_regression": joblib.load("saved_models/linear.pkl")["pipeline"],
    "ridge_regression": joblib.load("saved_models/ridge.pkl")["pipeline"],
    "lasso_regression": joblib.load("saved_models/lasso.pkl")["pipeline"],
    "elasticnet_regression": joblib.load("saved_models/elasticnet.pkl")["pipeline"],
    "gradient_boosting": joblib.load("saved_models/gradient_boosting.pkl")["pipeline"],
    "random_forest": joblib.load("saved_models/random_forest.pkl")["pipeline"],
    "gradient_boosting_hybrid": joblib.load("saved_models/gradient_boosting_hybrid.pkl")["pipeline"],
}

FEATURE_COLUMNS = [
    "temporada", "anio", "mes", "hora", "feriado",
    "dia_semana", "dia_trabajo", "clima", "temperatura",
    "humedad", "velocidad_viento"
]

# Entrada esperada para /predict
class ScooterInput(BaseModel):
    fecha: str
    temporada: int
    anio: int
    mes: int
    hora: float
    feriado: int
    dia_semana: float
    dia_trabajo: int
    clima: int
    temperatura: float
    sensacion_termica: float
    humedad: float
    velocidad_viento: float
    u_casuales: int
    u_registrados: int

# Entrada extendida para /predict_con_modelo
class ScooterInputConModelo(ScooterInput):
    modelo: str  # requerido para esta ruta

# Inicializa FastAPI
app = FastAPI()

@app.get("/")
def home():
    return {"mensaje": "API de predicción de alquileres de scooters"}

# Ruta original
@app.post("/predict")
def predict(data: ScooterInput):
    input_df = pd.DataFrame([data.dict()])
    cleaned_df = cleaning(input_df)
    X = cleaned_df[FEATURE_COLUMNS]
    model = MODELOS["linear_regression"]  # modelo fijo
    pred_log = model.predict(X)[0]
    pred = np.expm1(pred_log)
    return {"prediccion_alquileres": round(pred, 2)}

# Ruta con opción de modelo
@app.post("/predict_con_modelo")
def predict_con_modelo(data: ScooterInputConModelo):
    modelo_seleccionado = data.modelo.lower()

    if modelo_seleccionado not in MODELOS:
        raise HTTPException(
            status_code=400,
            detail=f"Modelo '{modelo_seleccionado}' no es válido. Opciones válidas: {list(MODELOS.keys())}"
        )

    # Convertir y limpiar
    input_dict = data.dict()
    input_dict.pop("modelo")
    input_df = pd.DataFrame([input_dict])
    cleaned_df = cleaning(input_df)
    X = cleaned_df[FEATURE_COLUMNS]

    # Predecir
    modelo = MODELOS[modelo_seleccionado]
    pred_log = modelo.predict(X)[0]
    pred = np.expm1(pred_log)

    return {
        "modelo_usado": modelo_seleccionado,
        "prediccion_alquileres": round(pred, 2)
    }
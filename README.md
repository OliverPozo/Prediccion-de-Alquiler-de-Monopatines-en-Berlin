# 🛴 Scooter Rental Prediction in Berlin

This project aims to predict the number of scooters rented per hour in Berlin, Germany, using machine learning techniques. It involves building ML models, integrating them into a reproducible pipeline, 
and exposing the models through an API.

---

## 📋 Table of Contents

- [Repository Structure](#repository-structure)  
- [Project Overview](#project-overview)  
- [Set up and Installation](#set-up-and-installation)  
- [Usage](#usage)  
- [API Overview](#api-overview)  
- [Deployment](#deployment)

---

## 📦Repository Structure

- `config/`: The hyperparameters used and data config. 
- `data/`: Original dataset and processed data. 
- `notebooks/`: Jupyter notebooks for EDA, feature engineering, models performance visualization.
- `saved_models/`: Serialized trained models (`.pkl` format).
- `src/`: Python modules for data preprocessing, training, and evaluation.
- `api.py`: Code to serve the model through a REST API
- `main.py`: To run the project 
- `requirements.txt`: Project dependencies.
- `README.md`: This file.

---

## 📄Project Overview
The dataset contains **17,380 records** with hourly scooter rental data in Berlin. The main objective is to accurately predict the number of scooters rented per hour using a robust machine learning workflow.

### Key Project Steps

- **Exploratory Data Analysis (EDA):** Understand data distributions, trends, and anomalies.
- **Data Cleaning:** Handle missing values and outliers.
- **Feature Engineering:** Create and select relevant features to improve model performance.
- **ML Pipeline Development:** Build reproducible pipelines for training and evaluation.
- **Model Evaluation & Visualization:** Assess model performance using appropriate metrics and visualize results.
- **API Integration:** Expose trained models via a REST API.
- **Deployment:** Make the solution accessible through cloud deployment.

### Models Utilized

#### Linear Models
- **Linear Regression** (baseline)
- **Ridge Regression**
- **Lasso Regression**
- **ElasticNet Regression**

#### Tree-Based Models
- **Random Forest**
- **Gradient Boosting**
- **Hybrid Gradient Boosting** (combining Random Search and Grid Search for hyperparameter tuning)

---

## 🔧Set up and Installation 

Follow the steps below to set up the project on your local machine: 
  
  1. Clone the repository:
  
    ```bash
    git clone https://github.com/OliverPozo/Prediccion-de-Alquiler-de-Monopatines-en-Berlin.git
    cd Prediccion-de-Alquiler-de-Monopatines-en-Berlin
  
  2. Install Required Libraries
  Install the necesary dependencies:

    ```bash
    pip install -r requirements.txt

---

## 🚀Usage 

To run the pipeline and evaluate model performance, execute the following command in your terminal:

```bash
python main.py
```

This will train the models, run the evaluation steps, and display performance metrics.

---

## 📊API Overview

This project provides a FastAPI-based REST API for making scooter rental predictions. You can use the default model (Linear Regression) or specify a different model for your prediction.

### 🚦 Running the API

To start the API locally, run:

```bash
uvicorn api:app --reload
```

### 🔗 Available Endpoints

- **`/predict`**  
    Make a prediction using the default model (Linear Regression).

- **`/predict_con_modelo`**  
    Make a prediction by specifying the model you want to use.

See the usage examples below for details on request formats.
### 🛠️ How to Use the API

You can make predictions by sending data in JSON format to the API endpoints.

#### **1. Default Prediction (`/predict`)**

Send a POST request to `/predict` with the following JSON structure:

```json
{
    "fecha": "2011-01-01",
    "temporada": 1,
    "anio": 0,
    "mes": 1,
    "hora": 0.0,
    "feriado": 0,
    "dia_semana": 6.0,
    "dia_trabajo": 0,
    "clima": 1,
    "temperatura": 0.24,
    "sensacion_termica": 0.2879,
    "humedad": 0.81,
    "velocidad_viento": 0.0,
    "u_casuales": 3,
    "u_registrados": 13
}
```

This will use the default model (Linear Regression) for prediction.

---

#### **2. Select Model (`/predict_con_modelo`)**

To specify a model, send a POST request to `/predict_con_modelo` with the same fields as above, plus a `modelo` field:

```json
{
    "fecha": "2011-01-01",
    "temporada": 1,
    "anio": 0,
    "mes": 1,
    "hora": 0.0,
    "feriado": 0,
    "dia_semana": 6.0,
    "dia_trabajo": 0,
    "clima": 1,
    "temperatura": 0.24,
    "sensacion_termica": 0.2879,
    "humedad": 0.81,
    "velocidad_viento": 0.0,
    "u_casuales": 3,
    "u_registrados": 13,
    "modelo": "random_forest"
}
```

**Available model options:**
- `linear_regression`
- `ridge_regression`
- `lasso_regression`
- `elasticnet_regression`
- `random_forest`
- `gradient_boosting`
- `gradient_boosting_hybrid`

---


## ☁Deployment

The project is deployed on [Railway](https://web-production-9b1b8.up.railway.app/).  
You can access the live API and interact with it using the endpoints described above.

**How to Use the Deployed API:**

- **Interactive Documentation:**  
    Visit the [API docs](https://web-production-9b1b8.up.railway.app/docs) for an interactive Swagger UI.

- **Make Predictions:**  
    Send POST requests to `/predict` or `/predict_con_modelo` as shown in the usage examples.

- **Feature Parity:**  
    The deployed API provides the same functionality as the local version.

---

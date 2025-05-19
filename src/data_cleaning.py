# preprocessing.py
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
    
def cleaning(df):
    df = drop_unnused_columns(df)
    df = drop_duplicates(df)
    df = drop_collinear_columns(df)
    #df = convert_to_datetime(df)    
    df = handle_missing_values(df)
    df = handling_outliers(df)
    df.drop(columns=['fecha'], inplace=True, errors='ignore')
    return df

def drop_unnused_columns(df):
    df.drop(columns=['u_casuales', 'u_registrados', 'indice'], inplace=True, errors='ignore')
    return df

def drop_duplicates(df):
    df.drop_duplicates(inplace=True)
    return df

def drop_collinear_columns(df):
    # Eliminar columnas colineales
    collinear_columns = ['sensacion_termica']
    df.drop(columns=collinear_columns, inplace=True, errors='ignore')
    return df

def convert_to_datetime(df):
    df['fecha'] = pd.to_datetime(df['fecha'])
    return df

def handle_missing_values(df):
    df.dropna(subset=['total_alquileres'], inplace=True) if 'total_alquileres' in df.columns else None
    mode_imputer = SimpleImputer(strategy='most_frequent')
    df[['hora', 'dia_semana']] = mode_imputer.fit_transform(df[['hora', 'dia_semana']])
    return df

def handling_outliers(df):
    # Para 'velocidad_viento'
    mean_wind = df[df['velocidad_viento'] <= 1]['velocidad_viento'].mean() 
    df.loc[df['velocidad_viento'] > 1, 'velocidad_viento'] = mean_wind     
    return df

def store_data(df, file_path):
    df.to_csv(file_path, index=False)

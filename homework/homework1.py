# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
import pandas as pd


# importo test
test_data = pd.read_csv(
    "files/input/test_data.csv.zip",
    index_col=False,
    compression="zip",
)
# importo train
train_data = pd.read_csv(
    "files/input/train_data.csv.zip",
    index_col=False,
    compression="zip",
)

#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
# - Renombre la columna "default payment next month" a "default".
test_data = test_data.rename(columns={"default payment next month": "default"})
train_data = train_data.rename(columns={"default payment next month": "default"})
# print(test_data.columns)

# - Remueva la columna "ID".
test_data = test_data.drop(columns=["ID"])
train_data = train_data.drop(columns=["ID"])

# - Elimine los registros con informacion no disponible.

import numpy as np

# Para train_data
train_data = train_data.loc[train_data["MARRIAGE"] != 0]
train_data = train_data.loc[train_data["EDUCATION"] != 0]
# print(train_data["EDUCATION"].value_counts())
# print(train_data["MARRIAGE"].value_counts())

# Para test_data (corregido)
test_data = test_data.loc[test_data["MARRIAGE"] != 0]
test_data = test_data.loc[test_data["EDUCATION"] != 0]
# print(test_data["EDUCATION"].value_counts())
# print(test_data["MARRIAGE"].value_counts())

# - Para la columna EDUCATION, valores > 4 indican niveles superiores
test_data["EDUCATION"] = test_data["EDUCATION"].apply(lambda x: 4 if x > 4 else x)
train_data["EDUCATION"] = train_data["EDUCATION"].apply(lambda x: 4 if x > 4 else x)

# print(train_data["EDUCATION"].value_counts())
# print(test_data["EDUCATION"].value_counts())

# Divida los datasets en x_train, y_train, x_test, y_test.
x_train = train_data.drop(columns="default")
y_train = train_data["default"]


x_test = test_data.drop(columns="default")
y_test = test_data["default"]


#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Ajusta un modelo de bosques aleatorios (rando forest).
#
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import f_classif

# fclassifier

# Columnas categoricas
categorical_features = ["SEX", "EDUCATION", "MARRIAGE"]

# preprocesador
preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(), categorical_features)],
    remainder="passthrough",
)

# pipeline
pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42)),
    ]
)


#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, balanced_accuracy_score


# Definir los hiperparámetros a optimizar
param_grid = {
    "classifier__n_estimators": [100],  # Número de árboles
    "classifier__max_depth": [None],  # Profundidad máxima
    "classifier__min_samples_split": [10],  # Mínimo de muestras para dividir
    "classifier__min_samples_leaf": [4],  # Mínimo de muestras por hoja
    "classifier__max_features": [23],
}

model = GridSearchCV(
    pipeline, param_grid, cv=10, scoring="balanced_accuracy", n_jobs=-1, refit=True
)

model.fit(x_train, y_train)


#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
import pickle
import os
import gzip


# Crear el directiorio para guardar el modelo
models_dir = "files/models"
os.makedirs(models_dir, exist_ok=True)

# Definir la ruta ruta del archivo comprimido
model_path = os.path.join(models_dir, "model.pkl.gz")

# Guardar el modelo comprimido
with gzip.open(model_path, "wb") as file:
    pickle.dump(model, file)


#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
import json
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
)


def calculate_and_save_metrics(model, X_train, X_test, y_train, y_test):
    # Hacer predicciones
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calcular métricas para el conjunto de entrenamiento
    metrics_train = {
        "type": "metrics",
        "dataset": "train",
        "precision": precision_score(y_train, y_train_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_train, y_train_pred),
        "recall": recall_score(y_train, y_train_pred, zero_division=0),
        "f1_score": f1_score(y_train, y_train_pred, zero_division=0),
    }

    # Calcular métricas para el conjunto de prueba
    metrics_test = {
        "type": "metrics",
        "dataset": "test",
        "precision": precision_score(y_test, y_test_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_test_pred),
        "recall": recall_score(y_test, y_test_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_test_pred, zero_division=0),
    }

    # Crear carpeta si no existe
    output_dir = "files/output"
    os.makedirs(output_dir, exist_ok=True)

    # Guardar las métricas en un archivo JSON
    output_path = os.path.join(output_dir, "metrics.json")
    with open(output_path, "w") as f:  # Usar 'w' para comenzar con un archivo limpio
        f.write(json.dumps(metrics_train) + "\n")
        f.write(json.dumps(metrics_test) + "\n")


#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
from sklearn.metrics import confusion_matrix


# Función para calcular las matrices de confusión y guardarlas
def calculate_and_save_confusion_matrices(model, X_train, X_test, y_train, y_test):
    # Hacer predicciones
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calcular matrices de confusión
    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_test = confusion_matrix(y_test, y_test_pred)

    # Convertir las matrices de confusión en formato JSON
    def format_confusion_matrix(cm, dataset_type):
        return {
            "type": "cm_matrix",
            "dataset": dataset_type,
            "true_0": {"predicted_0": int(cm[0, 0]), "predicted_1": int(cm[0, 1])},
            "true_1": {"predicted_0": int(cm[1, 0]), "predicted_1": int(cm[1, 1])},
        }

    metrics = [
        format_confusion_matrix(cm_train, "train"),
        format_confusion_matrix(cm_test, "test"),
    ]

    # Guardar las matrices de confusión en el mismo archivo JSON
    output_path = "files/output/metrics.json"
    with open(output_path, "a") as f:  # Usar 'a' para agregar después de las métricas
        for metric in metrics:
            f.write(json.dumps(metric) + "\n")


# Función principal para ejecutar todo
def main(model, X_train, X_test, y_train, y_test):
    # Crear el directorio de salida si no existe
    import os

    os.makedirs("files/output", exist_ok=True)

    # Calcular y guardar las métricas
    calculate_and_save_metrics(model, X_train, X_test, y_train, y_test)

    # Calcular y guardar las matrices de confusión
    calculate_and_save_confusion_matrices(model, X_train, X_test, y_train, y_test)


# Ejemplo de uso:
main(model, x_train, x_test, y_train, y_test)

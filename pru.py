
import pandas as pd

import joblib
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split  # Librería para guardar modelos de Machine Learning
ruta_modelo_fisicas = 'D://Aplicaciones//TFG2//Training//SavedModels_Supervisado//motor_fisicas_PolynomialFeatures.joblib'
modelo_fisicas = joblib.load(ruta_modelo_fisicas)

pepe = modelo_fisicas.predict([[40.1, 100, 0.0]])
print(pepe)

'''
df = pd.read_csv('D://Aplicaciones//TFG2//Datasets//telemetria_3Coches.csv')

X = df[['Speed', 'Throttle', 'Brake']].values
y = df['Delta_Speed'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)
print("Datos divididos en conjuntos de entrenamiento y prueba.")

jkj

# Hacer que el modelo prediga sobre el 20% de datos que NO ha visto
predicciones = modelo_fisicas.predict(X_test)

# Calcular el error medio absoluto
error = mean_absolute_error(y_test, predicciones)
print(f"Error medio del modelo: {error:.4f} km/h por tick")

'''
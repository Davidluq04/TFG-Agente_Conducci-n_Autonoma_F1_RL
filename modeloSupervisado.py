

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler  # Librería para guardar modelos de Machine Learning
from sklearn.pipeline import make_pipeline

# Leer un archivo CSV
df = pd.read_csv('D://Aplicaciones//TFG2//Datasets//telemetria_Vers2.csv')

X = df[['Speed', 'Throttle', 'Brake']].values
y = df['Delta_Speed'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Datos divididos en conjuntos de entrenamiento y prueba.")

#model = RandomForestRegressor(n_estimators=100, random_state=42)
model = RandomForestRegressor(n_estimators=10, max_depth=10, random_state=42)
#model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
#model = make_pipeline(StandardScaler(),
#    MLPRegressor(hidden_layer_sizes=(32, 16), activation='relu', max_iter=1000, random_state=42))
model.fit(X_train, y_train)
print("Modelo entrenado.")

# Hacer que el modelo prediga sobre el 20% de datos que NO ha visto
predicciones = model.predict(X_test)

# Calcular el error medio absoluto
error = mean_absolute_error(y_test, predicciones)
print(f"Error medio del modelo: {error:.4f} km/h por tick")


#ruta_guardado = 'D://Aplicaciones//TFG2//Training//SavedModels_Supervisado//motor_fisicas_RandomForestRegressor_v3.joblib'
#joblib.dump(model, ruta_guardado)
#print("¡Modelo guardado y listo para el simulador!")

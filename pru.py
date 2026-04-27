
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split  # Librería para guardar modelos de Machine Learning
ruta_modelo_fisicas = 'D://Aplicaciones//TFG2//Training//SavedModels_Supervisado//motor_fisicas_MLP_0.05seg.joblib'
modelo_fisicas = joblib.load(ruta_modelo_fisicas)
'''
pepe = modelo_fisicas.predict([[40.1, 100, 0.0]])
print(pepe)

'''
df = pd.read_csv('D://Aplicaciones//TFG2//Datasets//telemetria_Vers_0.05seg.csv')

X = df[['Speed', 'Throttle', 'Brake']].values
y = df['Delta_Speed'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)
print("Datos divididos en conjuntos de entrenamiento y prueba.")


# Hacer que el modelo prediga sobre el 20% de datos que NO ha visto
predicciones = modelo_fisicas.predict(X_test)

# Calcular el error medio absoluto
error = mean_absolute_error(y_test, predicciones)
print(f"Error medio del modelo: {error:.4f} km/h por tick")

# --- INICIO DEL CÓDIGO AÑADIDO PARA EVALUACIÓN ---

print("\n--- EVALUACIÓN DEL MODELO ---")

# 4. Cálculo de Métricas Numéricas
mae = mean_absolute_error(y_test, predicciones)
mse = mean_squared_error(y_test, predicciones)
rmse = np.sqrt(mse) # Raíz del error cuadrático medio
r2 = r2_score(y_test, predicciones)

print(f"Error Medio Absoluto (MAE): {mae:.4f} km/h por tick")
print(f"Error Cuadrático Medio (MSE): {mse:.4f}")
print(f"Raíz del MSE (RMSE): {rmse:.4f} km/h por tick (Penaliza errores grandes)")
print(f"Coeficiente de Determinación (R²): {r2:.4f} (El máximo es 1.0, valores < 0 son peores que predecir la media)")

# 5. Evaluación Visual (Gráficas)
# Gráfica 1: Predicciones vs Reales
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_test, predicciones, alpha=0.3, color='blue')
# Añadir la línea ideal (donde predicción = realidad)
min_val = min(np.min(y_test), np.min(predicciones))
max_val = max(np.max(y_test), np.max(predicciones))
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2)
plt.title('Valores Reales vs Predicciones')
plt.xlabel('Delta_Speed Real')
plt.ylabel('Delta_Speed Predicho')
plt.grid(True)

# Gráfica 2: Distribución de los residuos (Errores)
plt.subplot(1, 2, 2)
residuos = y_test - predicciones
plt.hist(residuos, bins=50, color='purple', edgecolor='black', alpha=0.7)
plt.axvline(x=0, color='red', linestyle='--', linewidth=2) # Línea de error cero
plt.title('Distribución de Residuos (Errores)')
plt.xlabel('Error (Real - Predicho)')
plt.ylabel('Frecuencia')
plt.grid(True)

plt.tight_layout()
plt.show()

# --- FIN DEL CÓDIGO AÑADIDO ---


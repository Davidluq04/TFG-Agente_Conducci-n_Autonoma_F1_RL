import pandas as pd
import numpy as np
from scipy.spatial import KDTree

# --- 1. PON AQUÍ LOS VALORES QUE TE DIO EL BOTÓN (Alineación final) ---
OFFSET_X = 142.04521535061917       # <-- Cámbialo por tu valor
OFFSET_Y = 100.77666825498295         # <-- Cámbialo por tu valor
ROTACION_GRADOS = 0.0  # <-- Cámbialo por tu valor
ESCALA = 0.1001271900897584           # <-- Cámbialo por tu valor

# --- 2. CARGAR DATOS ---
track = pd.read_csv('D://Aplicaciones//TFG2//Circuitos//Monza.csv')
telem = pd.read_csv('D://Aplicaciones//TFG2//Datasets//FuncionRecompensa//Monza_Telemetria_Ideal_2m.csv').dropna(subset=['X', 'Y', 'Speed'])

# --- 3. APLICAR TRANSFORMACIÓN A LA TELEMETRÍA ---
ang_rad = np.radians(ROTACION_GRADOS)

# Escalar
x_temp = telem['X'].values * ESCALA
y_temp = telem['Y'].values * ESCALA

# Rotar
x_rot = x_temp * np.cos(ang_rad) - y_temp * np.sin(ang_rad)
y_rot = x_temp * np.sin(ang_rad) + y_temp * np.cos(ang_rad)

# Trasladar
telem['ideal_x_trans'] = x_rot + OFFSET_X
telem['ideal_y_trans'] = y_rot + OFFSET_Y

# --- 4. MAPEAR LA TELEMETRÍA AL CIRCUITO (KDTree) ---
print("Buscando los puntos ideales más cercanos para cada punto del circuito...")

# Creamos un árbol de búsqueda hiper-rápido con los puntos transformados de Verstappen
arbol_piloto = KDTree(np.c_[telem['ideal_x_trans'], telem['ideal_y_trans']])

# Puntos de tu circuito original (el eje de la pista)
puntos_circuito = np.c_[track['x_m'], track['y_m']]

# Para cada punto de tu circuito, buscamos el punto del piloto más cercano
distancias, indices_cercanos = arbol_piloto.query(puntos_circuito)

# Asignamos la velocidad y la trazada ideal a las filas de tu circuito original
track['ideal_speed'] = telem['Speed'].iloc[indices_cercanos].values
track['ideal_x'] = telem['ideal_x_trans'].iloc[indices_cercanos].values
track['ideal_y'] = telem['ideal_y_trans'].iloc[indices_cercanos].values

# --- 5. GUARDAR EL CSV DEFINITIVO ---
nombre_salida = 'D://Aplicaciones//TFG2//Datasets//FuncionRecompensa//Monza_RL.csv'
track.to_csv(nombre_salida, index=False)

print(f"¡Hecho! '{nombre_salida}' generado correctamente con la telemetría inyectada.")
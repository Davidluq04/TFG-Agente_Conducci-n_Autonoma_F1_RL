import fastf1
import pandas as pd
import numpy as np

session = fastf1.get_session(2025, 'Monza', 'Q')
session.load()

laps = session.laps.pick_driver('VER').pick_fastest()
telemetry = laps.get_telemetry()

datos = {
    'Distance': telemetry['Distance'],
    'X': telemetry['X'],
    'Y': telemetry['Y'],
    'Speed': telemetry['Speed'],
}

df = pd.DataFrame(datos)

# CORRECCIÓN 1: Aplicar set_index sin reasignar la variable
df.set_index('Distance', inplace=True)

# CORRECCIÓN 2: Obtener la distancia máxima desde el índice
dist_max = df.index[-1]

# CORRECCIÓN 3: Usar np.arange para crear rangos con decimales (0.1)
nueva_distancia = np.arange(0, dist_max, 2)

# El resto de tu lógica es perfecta
indice_combinado = df.index.union(pd.Index(nueva_distancia)).drop_duplicates().sort_values()
df = df.reindex(indice_combinado)
df = df.interpolate(method='index')

df_final = df.reindex(nueva_distancia)
df_final.index.name = 'Distance'

# Opcional: Si quieres volver a tener 'Distance' como una columna normal y no como índice
df_final.reset_index(inplace=True)

print(df_final.head())




#Guardar en ruta local
ruta_guardado = './Datasets/FuncionRecompensa/Monza_Telemetria_Ideal_2m.csv'

# Guardamos el DataFrame resampleado (que ya contiene Speed, Throttle, Brake y Delta_Speed)
df_final.to_csv(ruta_guardado, index=False)

print(f"¡Hecho! CSV generado con {len(df_final)} puntos a intervalos exactos de 2 metros.")
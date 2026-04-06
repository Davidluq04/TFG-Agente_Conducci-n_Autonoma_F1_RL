import fastf1
import pandas as pd
import numpy as np


# 1. Cargar la sesión (Ej: Clasificación de Monza 2023)
session = fastf1.get_session(2023, 'Monza', 'Q')
session.load()

laps = session.laps.pick_driver('VER').pick_quicklaps()


nombrre_conductores = ['VER', 'LEC', 'SAI']

todos_df = []


for _, lap in laps.iterlaps():
    telemetry = lap.get_telemetry()

    # 3. Filtrar solo las columnas relevantes
    datos = {
        'Time': telemetry['Time'],
        'Speed': telemetry['Speed'],
        'Throttle': telemetry['Throttle'],
        'Brake': telemetry['Brake'],
    }

    # Convertir la columna 'Time' (que viene en formato timedelta) a segundos flotantes
    datos['Time'] = datos['Time'].dt.total_seconds()

    # Crear un nuevo vector de tiempo que vaya de 0.1 en 0.1 segundos
    tiempo_inicio = datos['Time'].min()
    tiempo_fin = datos['Time'].max()
    nuevo_tiempo = np.arange(tiempo_inicio, tiempo_fin, 0.1)

    df = pd.DataFrame(datos)

    df['Brake'] = df['Brake'].astype(float)  # Convertir a float para evitar problemas de interpolación

    df = df.set_index(datos['Time'])

    indice_combinado = df.index.union(nuevo_tiempo).drop_duplicates().sort_values()

    df = df.reindex(indice_combinado)

    df = df.interpolate(method='index')

    df_final = df.reindex(nuevo_tiempo)
    df_final.index.name = 'Time'

    df_final['Next_Speed'] = df_final['Speed'].shift(-1)

    #Obtener la y, el delta velocidad
    df_final['Delta_Speed']= df_final['Next_Speed'] - df_final['Speed']

    # Eliminar la última fila que contiene un valor NaN en 'Delta_Speed'
    df_final = df_final.dropna()

    todos_df.append(df_final)


dataset_completo = pd.concat(todos_df, ignore_index=True)

#Guardar en ruta local
ruta_guardado = 'D://Aplicaciones//TFG2//Datasets//telemetria_completa3.csv'

# Guardamos el DataFrame resampleado (que ya contiene Speed, Throttle, Brake y Delta_Speed)
dataset_completo.to_csv(ruta_guardado, index=False)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import fastf1
from stable_baselines3 import PPO
from entorno import F1Env
import os

# ==========================================
# CONFIGURACIÓN GLOBAL DE TAMAÑO DE TEXTO
# ==========================================
plt.rcParams.update({
    'font.size': 18,           
    'axes.labelsize': 20,      
    'xtick.labelsize': 16,     
    'ytick.labelsize': 16,     
    'legend.fontsize': 18      
})

def comparar_telemetria_ia_vs_f1():
    carpeta_salida = "resultados_comparativa"
    os.makedirs(carpeta_salida, exist_ok=True) 

    # ==========================================
    # 1. OBTENER DATOS DEL AGENTE IA
    # ==========================================
    print("--- PARTE 1: Simulando vuelta del Agente IA ---")
    model_path = ".\\Training\\SavedModels\\showerPPO\\Mejores_Modelos\\best_model_15C11.zip"
    track_file_path = './Datasets/FuncionRecompensa/Monza_RL.csv'
    
    circuito = pd.read_csv(track_file_path)
    env = F1Env(circuito, tipo_fisicas=3, tipo_recompensa="V2")
    model = PPO.load(model_path, env=env)

    obs, info = env.reset()
    done = False
    
    telemetria_ia = {'x': [], 'y': [], 'velocidad': [], 'pedal': []}

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        telemetria_ia['x'].append(env.state['car_x_position'])
        telemetria_ia['y'].append(env.state['car_y_position'])
        telemetria_ia['velocidad'].append(env.state['speed'])
        telemetria_ia['pedal'].append(action[0])
        
        done = terminated or truncated

    df_ia = pd.DataFrame(telemetria_ia)
    
    # Calcular la distancia acumulada de la IA (Metros) para alinearla con la F1
    dx = np.diff(df_ia['x'], prepend=df_ia['x'].iloc[0])
    dy = np.diff(df_ia['y'], prepend=df_ia['y'].iloc[0])
    df_ia['distancia'] = np.cumsum(np.sqrt(dx**2 + dy**2))

    # ==========================================
    # 2. OBTENER DATOS DE MAX VERSTAPPEN
    # ==========================================
    print("--- PARTE 2: Descargando telemetría de Max Verstappen ---")
    session = fastf1.get_session(2025, 'Monza', 'Q')
    session.load(telemetry=True)
    laps_ver = session.laps.pick_driver('VER')
    fastest_lap = laps_ver.pick_fastest()
    telemetry = fastest_lap.get_telemetry()

    dist_ver = telemetry['Distance']
    vel_ver = telemetry['Speed']

    # ==========================================
    # 3. CREAR GRÁFICA SUPERPUESTA
    # ==========================================
    print("--- PARTE 3: Generando gráfica vectorial ---")
    fig, axs = plt.subplots(2, 1, figsize=(16, 12), sharex=True)

    # --- SUBPLOT 1: VELOCIDAD SUPERPUESTA ---
    # Verstappen: Línea sólida azul
    axs[0].plot(dist_ver, vel_ver, color='blue', linewidth=2.5, label='Verstappen (F1 Real)')
    # IA: Línea discontinua naranja
    axs[0].plot(df_ia['distancia'], df_ia['velocidad'], color='orange', linewidth=2.5, linestyle='--', label='Agente IA')
    
    axs[0].set_ylabel('Velocidad (km/h)', fontsize=24, weight='bold')
    axs[0].grid(True, alpha=0.3)
    axs[0].set_ylim(0, max(vel_ver.max(), df_ia['velocidad'].max()) + 20)
    axs[0].legend(loc="lower left", fontsize=20)

    # --- SUBPLOT 2: PEDALES (SÓLO IA, ESTILO ORIGINAL) ---
    axs[1].fill_between(df_ia['distancia'], 0, df_ia['pedal'], where=(df_ia['pedal'] >= 0), color='green', alpha=0.5, label='Acelerador IA')
    axs[1].fill_between(df_ia['distancia'], 0, df_ia['pedal'], where=(df_ia['pedal'] < 0), color='red', alpha=0.5, label='Freno IA')
    axs[1].plot(df_ia['distancia'], df_ia['pedal'], color='black', linewidth=1)
    
    axs[1].set_ylabel('Uso Pedales\n(-1 a 1)', fontsize=24, weight='bold')
    axs[1].set_xlabel('Distancia del circuito (Metros)', fontsize=24, weight='bold')
    axs[1].axhline(0, color='black', linestyle='--')
    axs[1].grid(True, alpha=0.3)
    axs[1].legend(loc="upper right", fontsize=16)

    plt.tight_layout(pad=1.0, h_pad=2.0)
    ruta_pdf = os.path.join(carpeta_salida, "grafica_comparativa_ia_vs_verstappen.pdf")
    plt.savefig(ruta_pdf, format='pdf', bbox_inches='tight')
    plt.close()

    print(f"¡Éxito! Tu gráfica comparativa directa se ha guardado en: {ruta_pdf}")

if __name__ == '__main__':
    comparar_telemetria_ia_vs_f1()
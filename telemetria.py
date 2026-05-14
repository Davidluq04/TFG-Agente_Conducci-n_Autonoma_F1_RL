import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from entorno import F1Env

def generar_graficas_telemetria():
    # 1. CONFIGURACIÓN INICIAL Y CARGA DE MODELO
    model_path = "./Training/SavedModels/showerPPO/PPO_F1_1M_V4"
    track_file_path = 'D://Aplicaciones//TFG2//Circuitos//Monza.csv'
    
    print("Cargando entorno y modelo...")
    circuito = pd.read_csv(track_file_path)
    env = F1Env(circuito, tipo_fisicas=1, tipo_recompensa="V1")
    model = PPO.load(model_path, env=env)

    obs, info = env.reset()
    done = False

    # 2. DICCIONARIO PARA GUARDAR LA TELEMETRÍA (Con las nuevas variables)
    telemetria = {
        'x': [], 'y': [], 
        'velocidad': [], 
        'pedal': [], 
        'giro': [], 
        'step': [],
        'reward_instantaneo': [], 
        'reward_acumulado': [],   
        'jerk_volante': []        
    }

    print("Iniciando vuelta de recolección de datos (sin renderizado para ir más rápido)...")
    step = 0
    recompensa_total = 0
    giro_anterior = 0
    
    # 3. BUCLE DE SIMULACIÓN (Recolección de datos)
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        recompensa_total += reward
        cambio_giro = abs(action[1] - giro_anterior)
        
        # Guardamos el estado actual del coche y las acciones
        telemetria['x'].append(env.state['car_x_position'])
        telemetria['y'].append(env.state['car_y_position'])
        telemetria['velocidad'].append(env.state['speed'])
        telemetria['pedal'].append(action[0])
        telemetria['giro'].append(action[1])
        telemetria['step'].append(step)
        
        # Guardamos las métricas nuevas
        telemetria['reward_instantaneo'].append(reward)
        telemetria['reward_acumulado'].append(recompensa_total)
        telemetria['jerk_volante'].append(cambio_giro)
        
        giro_anterior = action[1]
        done = terminated or truncated
        step += 1

    print(f"Vuelta terminada. Datos recogidos: {step} ticks. Generando gráficas...")
    
    df = pd.DataFrame(telemetria)

    # ==========================================
    # GRÁFICA 1: MAPA DE LA TRAZADA Y VELOCIDAD
    # ==========================================
    plt.figure(figsize=(12, 8))
    plt.plot(env.pared_izq_x, env.pared_izq_y, color='black', linewidth=1, linestyle='--', label="Borde Izquierdo")
    plt.plot(env.pared_der_x, env.pared_der_y, color='black', linewidth=1, linestyle='--', label="Borde Derecho")
    scatter = plt.scatter(df['x'], df['y'], c=df['velocidad'], cmap='jet', s=15, zorder=5)
    cbar = plt.colorbar(scatter)
    cbar.set_label('Velocidad (km/h)', fontsize=12)
    plt.title('Trazada del Agente en Monza (Heatmap de Velocidad)', fontsize=16)
    plt.xlabel('Coordenada X (m)')
    plt.ylabel('Coordenada Y (m)')
    plt.axis('equal')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ==========================================
    # GRÁFICA 2: DASHBOARD DE TELEMETRÍA (F1 Style)
    # ==========================================
    fig, axs = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle('Telemetría de la IA (Velocidad, Pedales y Volante)', fontsize=18)
    axs[0].plot(df['step'], df['velocidad'], color='blue', linewidth=2)
    axs[0].set_ylabel('Velocidad (km/h)', fontsize=12)
    axs[0].grid(True, alpha=0.3)
    axs[0].set_ylim(0, 360)
    axs[1].fill_between(df['step'], 0, df['pedal'], where=(df['pedal'] >= 0), color='green', alpha=0.5, label='Acelerador')
    axs[1].fill_between(df['step'], 0, df['pedal'], where=(df['pedal'] < 0), color='red', alpha=0.5, label='Freno')
    axs[1].plot(df['step'], df['pedal'], color='black', linewidth=1)
    axs[1].set_ylabel('Uso Pedales (-1 a 1)', fontsize=12)
    axs[1].axhline(0, color='black', linestyle='--')
    axs[1].grid(True, alpha=0.3)
    axs[1].legend(loc="upper right")
    axs[2].plot(df['step'], df['giro'], color='purple', linewidth=2)
    axs[2].set_ylabel('Giro Volante (-1 a 1)', fontsize=12)
    axs[2].set_xlabel('Ticks de simulación (Tiempo)', fontsize=12)
    axs[2].axhline(0, color='black', linestyle='--')
    axs[2].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ==========================================
    # GRÁFICA 3: MAPA DE ACCIONES (Tu original)
    # ==========================================
    plt.figure(figsize=(8, 8))
    plt.scatter(df['giro'], df['pedal'], c=df['velocidad'], cmap='plasma', alpha=0.6)
    plt.axhline(0, color='black', linestyle='--')
    plt.axvline(0, color='black', linestyle='--')
    plt.title('Dispersión de Acciones: ¿Gira y frena a la vez?', fontsize=14)
    plt.xlabel('Volante (Izquierda < 0 > Derecha)')
    plt.ylabel('Pedales (Freno < 0 > Acelerador)')
    cbar = plt.colorbar()
    cbar.set_label('Velocidad (km/h)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ==========================================
    # GRÁFICA 4: ANÁLISIS DE RECOMPENSA Y SUAVIDAD
    # ==========================================
    fig, axs = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle('Evaluación del Rendimiento: Recompensa y Estabilidad de Conducción', fontsize=18)

    axs[0].plot(df['step'], df['reward_instantaneo'], color='darkorange', linewidth=1.5)
    axs[0].axhline(0, color='black', linestyle='--')
    axs[0].set_ylabel('Recompensa por Tick', fontsize=12)
    axs[0].set_title('Recompensa obtenida en cada instante (Influenciada por avance y velocidad)', fontsize=10)
    axs[0].grid(True, alpha=0.3)

    axs[1].plot(df['step'], df['reward_acumulado'], color='teal', linewidth=2)
    axs[1].set_ylabel('Recompensa Total', fontsize=12)
    axs[1].set_title('Evolución de la Función de Recompensa a lo largo del circuito', fontsize=10)
    axs[1].grid(True, alpha=0.3)

    axs[2].plot(df['step'], df['jerk_volante'], color='crimson', linewidth=1.5)
    axs[2].set_ylabel('Brusquedad Volante (\u0394 Giro)', fontsize=12)
    axs[2].set_title('Derivada del giro: Picos altos indican "volantazos" erráticos de la red neuronal', fontsize=10)
    axs[2].set_xlabel('Ticks de simulación (Tiempo)', fontsize=12)
    axs[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    generar_graficas_telemetria()
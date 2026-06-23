import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

def generar_graficas_telemetria():
    # 1. CONFIGURACIÓN INICIAL Y CARGA DE MODELO
    #model_path = "./Training/SavedModels/showerPPO/PPO_F1_5M_V15_C4.zip"
    model_path = ".\\Training\\Modelos_Finales\\best_model_15C11.zip"
    track_file_path = './Datasets/FuncionRecompensa/Monza_RL.csv'
    
    print("Cargando entorno y modelo...")
    circuito = pd.read_csv(track_file_path)
    env = F1Env(circuito, tipo_fisicas=3, tipo_recompensa="V2")
    model = PPO.load(model_path, env=env)

    obs, info = env.reset()
    done = False

    # 2. DICCIONARIO PARA GUARDAR LA TELEMETRÍA
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
    
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        recompensa_total += reward
        cambio_giro = abs(action[1] - giro_anterior)
        
        telemetria['x'].append(env.state['car_x_position'])
        telemetria['y'].append(env.state['car_y_position'])
        telemetria['velocidad'].append(env.state['speed'])
        telemetria['pedal'].append(action[0])
        telemetria['giro'].append(action[1])
        telemetria['step'].append(step)
        
        telemetria['reward_instantaneo'].append(reward)
        telemetria['reward_acumulado'].append(recompensa_total)
        telemetria['jerk_volante'].append(cambio_giro)
        
        giro_anterior = action[1]
        done = terminated or truncated
        step += 1

    print(f"Vuelta terminada. Datos recogidos: {step} ticks.")
    
    df = pd.DataFrame(telemetria)

    # CREAR CARPETA Y GUARDAR DATOS EN CSV
    carpeta_salida = "resultados_telemetria"
    os.makedirs(carpeta_salida, exist_ok=True) 
    
    ruta_csv = os.path.join(carpeta_salida, "telemetria_monza.csv")
    df.to_csv(ruta_csv, index=False)
    print("Generando y guardando gráficas en formato PDF vectorial...")

    # ==========================================
    # GRÁFICA 1: MAPA DE LA TRAZADA Y VELOCIDAD (SIN LEYENDA)
    # ==========================================
    plt.figure(figsize=(14, 10))
    plt.plot(env.pared_izq_x, env.pared_izq_y, color='black', linewidth=1, linestyle='--') # Quitamos los labels
    plt.plot(env.pared_der_x, env.pared_der_y, color='black', linewidth=1, linestyle='--')
    scatter = plt.scatter(df['x'], df['y'], c=df['velocidad'], cmap='jet', s=15, zorder=5)
    
    cbar = plt.colorbar(scatter)
    cbar.set_label('Velocidad (km/h)', fontsize=28, weight='bold')
    cbar.ax.tick_params(labelsize=22) 
    
    plt.xlabel('Coordenada X (m)', fontsize=28, weight='bold')
    plt.ylabel('Coordenada Y (m)', fontsize=28, weight='bold')
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    
    plt.axis('equal')
    # plt.legend() ELIMINADO para quitar el mensaje de "borde izquierdo/derecho"
    plt.tight_layout(pad=1.0) 
    plt.savefig(os.path.join(carpeta_salida, "grafica1_trazada.pdf"), format='pdf', bbox_inches='tight')
    plt.close()

    # ==========================================
    # GRÁFICA 2: DASHBOARD DE TELEMETRÍA (F1 Style)
    # ==========================================
    fig, axs = plt.subplots(3, 1, figsize=(16, 14), sharex=True)
    
    axs[0].plot(df['step'], df['velocidad'], color='blue', linewidth=2)
    axs[0].set_ylabel('Velocidad (km/h)', fontsize=20)
    axs[0].grid(True, alpha=0.3)
    axs[0].set_ylim(0, 360)
    
    axs[1].fill_between(df['step'], 0, df['pedal'], where=(df['pedal'] >= 0), color='green', alpha=0.5, label='Acelerador')
    axs[1].fill_between(df['step'], 0, df['pedal'], where=(df['pedal'] < 0), color='red', alpha=0.5, label='Freno')
    axs[1].plot(df['step'], df['pedal'], color='black', linewidth=1)
    axs[1].set_ylabel('Uso Pedales\n(-1 a 1)', fontsize=20)
    axs[1].axhline(0, color='black', linestyle='--')
    axs[1].grid(True, alpha=0.3)
    axs[1].legend(loc="upper right")
    
    axs[2].plot(df['step'], df['giro'], color='purple', linewidth=2)
    axs[2].set_ylabel('Giro Volante\n(-1 a 1)', fontsize=20)
    axs[2].set_xlabel('Ticks de simulación (Tiempo)', fontsize=20)
    axs[2].axhline(0, color='black', linestyle='--')
    axs[2].grid(True, alpha=0.3)
    
    plt.tight_layout(pad=1.0, h_pad=2.0)
    plt.savefig(os.path.join(carpeta_salida, "grafica2_dashboard.pdf"), format='pdf', bbox_inches='tight')
    plt.close()

    # ==========================================
    # GRÁFICA 3: MAPA DE ACCIONES
    # ==========================================
    plt.figure(figsize=(10, 10))
    plt.scatter(df['giro'], df['pedal'], c=df['velocidad'], cmap='plasma', alpha=0.6)
    plt.axhline(0, color='black', linestyle='--')
    plt.axvline(0, color='black', linestyle='--')
    
    plt.xlabel('Volante (Izquierda < 0 > Derecha)')
    plt.ylabel('Pedales (Freno < 0 > Acelerador)')
    cbar = plt.colorbar()
    cbar.set_label('Velocidad (km/h)', fontsize=20)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout(pad=1.0)
    plt.savefig(os.path.join(carpeta_salida, "grafica3_acciones.pdf"), format='pdf', bbox_inches='tight')
    plt.close()

    # ==========================================
    # GRÁFICA 4: ANÁLISIS DE RECOMPENSA Y SUAVIDAD
    # ==========================================
    fig, axs = plt.subplots(3, 1, figsize=(16, 14), sharex=True)

    axs[0].plot(df['step'], df['reward_instantaneo'], color='darkorange', linewidth=1.5)
    axs[0].axhline(0, color='black', linestyle='--')
    axs[0].set_ylabel('Recompensa\npor Tick', fontsize=20)
    axs[0].grid(True, alpha=0.3)

    axs[1].plot(df['step'], df['reward_acumulado'], color='teal', linewidth=2)
    axs[1].set_ylabel('Recompensa\nTotal', fontsize=20)
    axs[1].grid(True, alpha=0.3)

    axs[2].plot(df['step'], df['jerk_volante'], color='crimson', linewidth=1.5)
    axs[2].set_ylabel('Brusquedad\n(Δ Giro)', fontsize=20)
    axs[2].set_xlabel('Ticks de simulación (Tiempo)', fontsize=20)
    axs[2].grid(True, alpha=0.3)

    plt.tight_layout(pad=1.0, h_pad=2.0)
    plt.savefig(os.path.join(carpeta_salida, "grafica4_rendimiento.pdf"), format='pdf', bbox_inches='tight')
    plt.close()

    # ==========================================
    # TABLA 5: RESUMEN DE LA VUELTA Y MÉTRICAS (TAMAÑO REDUCIDO CON CABECERAS)
    # ==========================================
    tiempo_total_segundos = len(df) * 0.1
    minutos = int(tiempo_total_segundos // 60)
    segundos = tiempo_total_segundos % 60
    tiempo_formateado = f"{minutos:02d}:{segundos:05.2f}"
    
    recompensa_final = df['reward_acumulado'].iloc[-1]
    velocidad_media = df['velocidad'].mean()
    velocidad_maxima = df['velocidad'].max()
    porcentaje_acelerador = (df['pedal'] > 0).mean() * 100
    porcentaje_freno = (df['pedal'] < 0).mean() * 100
    
    datos_tabla = [
        ['Tiempo por vuelta', tiempo_formateado],
        ['Recompensa Total', f"{recompensa_final:.2f} pts"],
        ['Velocidad Media', f"{velocidad_media:.2f} km/h"],
        ['Velocidad Máxima', f"{velocidad_maxima:.2f} km/h"],
        ['% Tiempo Acelerando', f"{porcentaje_acelerador:.1f} %"],
        ['% Tiempo Frenando', f"{porcentaje_freno:.1f} %"]
    ]
    
    # Hemos devuelto las cabeceras
    columnas = ['Métrica', 'Valor']

    fig_tabla, ax_tabla = plt.subplots(figsize=(12, 6))
    ax_tabla.axis('off')
    
    tabla = ax_tabla.table(cellText=datos_tabla, 
                           colLabels=columnas, # Se añaden las columnas aquí
                           cellLoc='center', 
                           loc='center',
                           bbox=[0, 0, 1, 1]) 
    
    tabla.auto_set_font_size(False)
    # Letra un pelín más pequeña (26 en vez de 34)
    tabla.set_fontsize(26)
    
    for (row, col), cell in tabla.get_celld().items():
        if row == 0:
            # Estilo para la cabecera devuelta
            cell.set_text_props(weight='bold', color='white', fontsize=28)
            cell.set_facecolor('#4CAF50')
        else:
            if row % 2 == 0:
                cell.set_facecolor('#ffffff')
            else:
                cell.set_facecolor('#f8f9fa')
                
            if col == 0:
                cell.set_text_props(weight='bold', color='#333333')

    plt.savefig(os.path.join(carpeta_salida, "grafica5_resumen_tabla.pdf"), format='pdf', bbox_inches='tight')
    plt.close()

    print(f"¡Listo! Revisa la carpeta '{carpeta_salida}'.")

if __name__ == '__main__':
    generar_graficas_telemetria()
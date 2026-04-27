import pandas as pd

from stable_baselines3 import PPO

from entorno import F1Env


def testear_modelo():
    # Ruta al modelo guardado
    model_path = "./Training/SavedModels/showerPPO/PPO_F1_5M_V6"

    #CREACION DEL AGENTE
    track_file_path = 'D://Aplicaciones//TFG2//Circuitos//Monza.csv'
    circuito = pd.read_csv(track_file_path)
    env = F1Env(circuito)

    # 🚀 CARGAR EL MODELO ENTRENADO
    # Observa que usamos PPO.load() en lugar de PPO('MultiInputPolicy', ...)
    model = PPO.load(model_path, env=env)

    print("Modelo cargado correctamente. Iniciando examen...")

    print("\n--- EXAMEN DE CONDUCIR ---")
    obs, info = env.reset()
    done = False
    total_reward = 0
    env.render()

    while not done:
        
        # 🌟 LA MAGIA ESTÁ AQUÍ 🌟
        # Le pasamos lo que ven los radares y el velocímetro a la IA, y ella decide los pedales
        action, _states = model.predict(obs, deterministic=True) 
        
        # Le pasamos la decisión de la IA al simulador
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        env.render()


    print(f"Resultado final del examen: Premio = {total_reward:.1f}")


if __name__ == '__main__':
    testear_modelo() # (o testear_modelo_entrenado() si usaste la primera versión)
import pandas as pd

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


from entorno import F1Env


def testear_modelo():
    track_file_path = 'D://Aplicaciones//TFG2//Circuitos//Monza.csv'
    model_path = "./Training/SavedModels/showerPPO/Mejores_Modelos/best_model.zip"
    # Ruta al modelo guardado
    circuito = pd.read_csv(track_file_path)
    env = F1Env(circuito, tipo_fisicas=3, tipo_recompensa="V1")

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
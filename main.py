import pandas as pd

import pygame
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


from entorno import F1Env


def testear_modelo(track_file_path, model_path, tipo_fisicas, tipo_recompensa):
    if "Básicas" in tipo_fisicas:
        fisicas_int = 1
    elif "Supervisadas" in tipo_fisicas:
        fisicas_int = 2
    else:
        fisicas_int = 3 # Avanzadas

    if "V1" in tipo_recompensa:
        recompensa_str = "V1"
    else:
        recompensa_str = "V2"

    circuito = pd.read_csv(track_file_path)
    env = F1Env(circuito, tipo_fisicas=fisicas_int, tipo_recompensa=recompensa_str)


    
    model = PPO.load(model_path, env=env)

    print("Modelo cargado correctamente.")

    obs, info = env.reset()
    done = False
    total_reward = 0
    

    while not done:
        try:
            comando = env.render()

            if comando == "QUIT":
                print("Examen interrumpido por el usuario.")
                break
                
            elif comando == "RESTART":
                print("Reiniciando el examen...")
                obs, info = env.reset()
                total_reward = 0
                continue
            elif comando == "PAUSE":
                continue
        
        # Le pasamos lo que ven los radares y el velocímetro a la IA, y ella decide los pedales
            action, _states = model.predict(obs, deterministic=True) 
            
            # Le pasamos la decisión de la IA al simulador
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            
        except Exception as e:
            print(f"Error durante el examen: {e}")
            break


    print(f"Resultado final del examen: Premio = {total_reward:.1f}")

    try:
        pygame.quit()
    except:
        pass


if __name__ == '__main__':
    testear_modelo()
#GYM
import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete
#Box te permite un rango de valores, en un coche es lo que nos hace falta

#General
import os
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
#import fastF1

#Stable-Baselines
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

import pygame
from pygame.locals import *

import joblib

from f1_render import F1Renderer  # Librería para guardar modelos de Machine Learning

class F1Env(Env):
  def __init__(self, track_file_path):

    self.track_data = track_file_path
    self.track_data_len = len(self.track_data)

    self.dt = 0.1

    #OBTENER DISTANCIA TOTAL DEL CIRCUTIO PARA LUEGO EL STEP
    self.x = self.track_data['x_m'].values
    self.y = self.track_data['y_m'].values
    self.ancho_der = self.track_data['w_tr_right_m'].values
    self.ancho_izq = self.track_data['w_tr_left_m'].values

    x_diff = np.diff(self.x)
    y_diff = np.diff(self.y)
    segment_lengths = np.sqrt(x_diff**2 + y_diff**2)
    self.track_length = np.sum(segment_lengths)


    #PYGAME
    self.renderer = None


    #OBTENER LAS PAREDES DEL FINAL DE PISTA
    
    dx = np.diff(self.x, append=self.x[0])  # Diferencia en x, con append para cerrar el circuito
    dy = np.diff(self.y, append=self.y[0])  # Diferencia en y, con append para cerrar el circuito

    segment_lengths = np.sqrt(dx**2 + dy**2)  # Longitud de cada segmento

    nx = -dy / segment_lengths  # Vector normal en x
    ny = dx / segment_lengths   # Vector normal en y


    self.pared_izq_x = self.x + nx * self.ancho_izq
    self.pared_izq_y = self.y + ny * self.ancho_izq

    self.pared_der_x = self.x - nx * self.ancho_der
    self.pared_der_y = self.y - ny * self.ancho_der

    #CARGAR EL MODELO DE FISICAS
    ruta_modelo_fisicas = 'D://Aplicaciones//TFG2//Training//SavedModels_Supervisado//motor_fisicas_PolynomialFeatures.joblib'


    if not os.path.exists(ruta_modelo_fisicas):
        raise FileNotFoundError(f"No se encontró el modelo en: {ruta_modelo_fisicas}")
    
    self.modelo_fisicas = joblib.load(ruta_modelo_fisicas)


    #AÑADIR VUELTAS SI NO QUIERO QUE SOLO HAGA UNA VUELTA TODO EL RATO


    self.action_space = Box(low = -1, high = 1, shape = (2,)) #Lo que realiza el agente, shape 2 para que sea volante y pedales

    #Lo que se le pasa al agente en cada momento
    self.observation_space = Dict({
        'speed': Box(low = 0, high = 1.0, shape = (1,)),
        #'position': Box(low = 0, high = 1.0, shape = (1,)),
        'radars': Box(low = 0.0, high = 1.0, shape = (11,)),
        'delta_angulo': Box(low = -1, high = 1, shape = (1,))
        })

    #Estado agente
    self.state = {
        'speed': 0,
        'position': 0,
        'car_x_position': 0,
        'car_y_position': 0,
        'angulo':0,
        #Añadimos esto para no ir calculando en cada step la posicion del coche en el CSV, ya que es un calculo pesado, asi que lo guardamos y solo lo actualizamos cada vez que el coche avanza
        'idx_csv': 0,
        'pedal': 0 #para dibujar la barra
    }
    #Radares los vamos a calcular en cada step, no los guardamos en el estado porque no los necesitamos para nada mas que para la observacion, y asi no tenemos que preocuparnos de actualizarlos cada vez que el coche se mueve

  def step(self, action):

    #Recibimos la accion que ha hecho el coche
    pedal = action[0]
    giro = action[1]

    self.state['pedal'] = pedal # Guardamos el valor del pedal para dibujar la barra

    #-----------------------------CAMBIAMOS VELOCIDAD-----------------------
    '''
    self.state['speed'] = self.calcular_velocidad_fisicas_basica(pedal)
    '''

    '''
    self.state['speed'] = self.calcular_velocidad_fisicas_Supervisado(pedal, dt=self.dt)
    '''

    self.state['speed'] = self.calcular_velocidad_fisicas(pedal, dt=self.dt)

    
    #---------------------------CAMBIAMOS POSICION COCHE------------------------

    '''
    self.calcular_posicion_coche_basico(giro)
    '''

    self.calcular_posicion_coche(giro)


    
   
    #-------------------------FUNCION DE RECOMPENSA--------------------------
    '''
    La funcion de recompensa la he cambiado del dia 2 al 4, en el dia 2 calculaba el idx en base
    a la posicion en porcentaje y eso podia hacer que variase y no fuese exacto, por eso en el dia 4
    he puesto lo de abajo, para ello, he añadido en state un idx_anterior_tick para ir calculando el idx
    constantemente. Tiempo despues He cambiado ahora lo inicio fin para hacer que cuando llegue
    al final del circuito pueda volver a ver los puntos del inicio, para que no se quede sin puntos cercanos
    y el coche no se vuelva loco, ademas de que asi puede hacer varias vueltas sin problemas. 
    '''


    reward, terminated, avanzado_metros = self.calcular_recompensa() 
    #---------------------CALCULAR POSITION, INDICE, DELTA_ANGULO----------------------------

    metros_por_punto = self.track_length / self.track_data_len
    tiempo_vision = 1.5
    velcidad_ms = self.state['speed'] / 3.6

    metro_delante = max(10, velcidad_ms * tiempo_vision) # Distancia que recorre en 1.5 segundo a la velocidad actual, para anticiparse a las curvas
    
    puntos_delante = int(metro_delante / metros_por_punto) # Convertimos esa distancia a puntos del CSV

    idx_futuro = (self.idx_real + puntos_delante) % self.track_data_len #punto en el futuro para anticiparse a las curvas, es como los radares pero para curva
    
    x_delante = self.x[idx_futuro]
    y_delante = self.y[idx_futuro]
    x_actual = self.x[self.idx_real]
    y_actual = self.y[self.idx_real]

    angulo_relativo = np.arctan2(y_delante - y_actual, x_delante - x_actual)
    delta_angulo = (angulo_relativo - self.state['angulo'] + np.pi) % (2 * np.pi) - np.pi # Normalizamos el delta de ángulo a un rango de -pi a pi para que la IA lo entienda mejor


    
    por_avanzado = (avanzado_metros/self.track_length) * 100
    self.state['position'] += por_avanzado
    self.state['idx_csv'] = self.idx_real # Guardamos el índice real para el siguiente tick

    
    #---------------------CALCULAR RADARES----------------------------

    radares = self.calculo_radares()
    #Lo normalizamos a un rango de 0 a 1 para que la IA lo entienda mejor, ya que el rango de los radares es de 0 a 300 metros, dividimos entre 300 para que el valor máximo sea 1 y el mínimo 0
    radares_normalizados = [x / 300 for x in radares]
           

    #---------------------CALCULAR OBSERVATION----------------------------

    obs = {
        'speed': np.array([self.state['speed'] / 340], dtype=np.float32),
        #'position': np.array([self.state['position'] % 100 / 100], dtype=np.float32),
        'radars': np.array(radares_normalizados, dtype=np.float32),
        'delta_angulo': np.array([delta_angulo / np.pi], dtype=np.float32) # Normalizamos el delta de ángulo a un rango de -1 a 1 dividiendo entre pi, asi la IA lo entiende mejor
    }


    return obs, reward, terminated, False, {}





  def render(self):
    
    # Cargamos el motor gráfico solo si alguien llama a env.render()
    if self.renderer is None:
        self.renderer = F1Renderer(
            self.pared_izq_x, self.pared_izq_y, 
            self.pared_der_x, self.pared_der_y
        )
    # Le pasamos el estado actual para que lo dibuje
    self.renderer.render(self.state)






  def reset(self, seed=None, options=None):
    super().reset(seed=seed)

    #Estado inicial del coche

    # Calcular la dirección del primer segmento de la pista
    self.dx_init = self.x[1] - self.x[0]
    self.dy_init = self.y[1] - self.y[0]
    self.state['angulo'] = np.arctan2(self.dy_init, self.dx_init) # Orienta el coche hacia adelante

   
    self.state['speed'] = 0
    self.state['position'] = 0
    self.state['car_x_position'] = self.x[0] # Empezamos en la posición del primer punto del CSV
    self.state['car_y_position'] = self.y[0]
    self.state['idx_csv'] = 0
    self.state['pedal'] = 0


    #La observacion inicial
    obs = {
        'speed': np.array([self.state['speed'] / 340], dtype=np.float32),
        #'position': np.array([self.state['position'] % 100 / 100], dtype=np.float32),
        'radars': np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32),
        'delta_angulo': np.array([0], dtype=np.float32)
    }

    return obs, {}
  




  #===========================================================
  #                   CALCULO RADARES
  #===========================================================

  def calculo_radares(self):

    radars = []
    angulo_pri = self.state['angulo']

    # 1. Calculamos cuántos metros reales representa cada fila del CSV
    metros_por_punto = self.track_length / self.track_data_len
    
    # 2. Calculamos cuántos puntos necesitamos para cubrir 320m atrás y 320 adelante
    puntos_atras = int(320 / metros_por_punto)
    puntos_adelante = int(320 / metros_por_punto) # 320 para que cubra los 300m del radar de sobra

    idx_tick = self.state['idx_csv']
    
    # 3. Recortamos la pista con esas medidas exactas 
    indice = np.arange(idx_tick - puntos_atras, idx_tick + puntos_adelante) % self.track_data_len

    indice_sig = (indice + 1) % self.track_data_len

    #El radar esta con el coche y el area que ve
    #alrededor suya coincide con la posicion actual que ve ahora mismo a diferencia de antes
    #en la funcion de recompensa que tenemos que trabajar al principio con el indice anterior
    #aunque no influya porque tenemos la posicion del coche

    '''    
    pista_x = self.x[indice]
    pista_y = self.y[indice]
    pista_izq = self.ancho_izq[indice]
    pista_der = self.ancho_der[indice]
    '''

    x_radar_ini = self.state['car_x_position']
    y_radar_ini = self.state['car_y_position']

    #obtener paredes coche
    x3_pared_izq, y3_pared_izq = self.pared_izq_x[indice], self.pared_izq_y[indice]
    x4_pared_izq, y4_pared_izq = self.pared_izq_x[indice_sig], self.pared_izq_y[indice_sig]

    x3_pared_der, y3_pared_der = self.pared_der_x[indice], self.pared_der_y[indice]
    x4_pared_der, y4_pared_der = self.pared_der_x[indice_sig], self.pared_der_y[indice_sig]


    x3 = np.concatenate((x3_pared_izq, x3_pared_der))
    y3 = np.concatenate((y3_pared_izq, y3_pared_der))

    x4 = np.concatenate((x4_pared_izq, x4_pared_der))
    y4 = np.concatenate((y4_pared_izq, y4_pared_der))

    for i in range(11):
        angulor_ra = angulo_pri + np.radians(-90 + i*18) # Radares cada 45 grados, empezando por el de la izquierda

        x_radar_fin = self.state['car_x_position'] + 300 * np.cos(angulor_ra)
        y_radar_fin = self.state['car_y_position'] + 300 * np.sin(angulor_ra)

        denominador = (x_radar_ini - x_radar_fin) * (y3 - y4) - (y_radar_ini - y_radar_fin) * (x3 - x4)
        # Evitar división por cero
        denominador[denominador == 0] = 1e-10

        t = ((x_radar_ini - x3) * (y3 - y4) - (y_radar_ini - y3) * (x3 - x4)) / denominador
        u = -((x_radar_ini - x_radar_fin) * (y_radar_ini - y3) - (y_radar_ini - y_radar_fin) * (x_radar_ini - x3)) / denominador

        # ¿Dónde hay colisiones válidas?
        colisiones = (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)

        if np.any(colisiones):
            # Si hay varias (ej: curvas en S), nos quedamos con la más cercana (el 't' más pequeño)
            distancia_choque = np.min(t[colisiones]) * 300.0
            radars.append(distancia_choque)
        else:
            # Si no se cruzó con ninguna pared, el radar llega a su máximo
            radars.append(300.0)

        '''
         dist = 0
        lim = False

        while lim == False and dist < 300:
            dist += 1
            x_radar = self.state['car_x_position'] + dist * np.cos(angulor_ra)
            y_radar = self.state['car_y_position'] + dist * np.sin(angulor_ra)

            #Calcular la posicion del radar en el CSV
            # Las raíces cuadradas (np.sqrt) son lentísimas de calcular.
            # Para saber cuál es el punto más cercano, no necesitamos la raíz, 
            # nos basta con comparar las distancias al cuadrado.
            distancias_cuadradas = (x_radar - pista_x)**2 + (y_radar - pista_y)**2
            #ESte cercano correponde al radar, el otro al coche
            idx_cercano_radar = np.argmin(distancias_cuadradas)
            idx_real_radar = indice[idx_cercano_radar]

            # Solo le hacemos la raíz cuadrada al punto ganador para tener la medida real
            distancia_al_centro = np.sqrt(distancias_cuadradas[idx_cercano_radar])

            #Aplicamos Producto Cruz para saber si el radar esta a la izq o der de la linea central
            x_central_radar = pista_x[idx_cercano_radar]
            y_central_radar = pista_y[idx_cercano_radar]


            #Calculamos la posicion siguiente asi para tratar el circuito como un bucle,
            #ya que si el radar esta cerca del final del circuito, el siguiente punto es el del inicio
            pos_csv_sig = (idx_real_radar + 1) % self.track_data_len
            
            x_csv_sig = self.x[pos_csv_sig]
            y_csv_sig = self.y[pos_csv_sig]

            pro_cr = (x_csv_sig-x_central_radar) * (y_radar - y_central_radar) - (y_csv_sig-y_central_radar) * (x_radar - x_central_radar)

            #Calcular si el radar esta fuera de la pista
            if  pro_cr > 0:
                # El radar está a la IZQUIERDA del centro
                lim = distancia_al_centro > pista_izq[idx_cercano_radar]
            else:
                # El radar está a la DERECHA del centro
                lim = distancia_al_centro > pista_der[idx_cercano_radar]  

        radars.append(dist)   
        '''
        
    return radars
  




  #===========================================================
  #                 CALCULO RECOMPENSA 
  #===========================================================

  
  def calcular_recompensa(self):
    coche_en_grava = False


    idx_anterior_tick = self.state['idx_csv']
    
    # 3. Recortamos la pista con esas medidas exactas

    indice = np.arange(idx_anterior_tick - 30, idx_anterior_tick + 30) % self.track_data_len

    x_csv = self.x[indice]
    y_csv = self.y[indice]

    der_csv = self.ancho_der[indice]
    izq_csv = self.ancho_izq[indice]

    posicion_co_x = self.state['car_x_position']
    posicion_co_y = self.state['car_y_position']
    


    #Obtenemos el punto del CSV más cercano al coche, para eso calculamos la distancia al cuadrado de cada punto del CSV con respecto a la posicion del coche, y nos quedamos con el indice del punto que tenga la distancia al cuadrado mas baja, ese es el punto del CSV mas cercano al coche
    distancias_cuadradas = (self.state['car_x_position'] - x_csv)**2 + (self.state['car_y_position'] - y_csv)**2
    idx_cercano = np.argmin(distancias_cuadradas)

    self.idx_real = indice[idx_cercano] 
    # Solo le hacemos la raíz cuadrada al punto ganador para tener la medida real
    distancia_al_centro = np.sqrt(distancias_cuadradas[idx_cercano])

    #Vamos a calcular si esta a la izq o der de la linea central
    #Aplicamos Producto Cruz
    x_punto_actual = x_csv[idx_cercano]
    y_punto_actual = y_csv[idx_cercano]


    pos_csv_sig = (self.idx_real + 1) % self.track_data_len
    x_csv_sig = self.x[pos_csv_sig]
    y_csv_sig = self.y[pos_csv_sig]

    pro_cr = (x_csv_sig-x_punto_actual) * (posicion_co_y - y_punto_actual) - (y_csv_sig-y_punto_actual) * (posicion_co_x - x_punto_actual)

    if pro_cr > 0:
        # El coche está a la IZQUIERDA del centro
        coche_en_grava = distancia_al_centro > izq_csv[idx_cercano]
    else:
    # El coche está a la DERECHA del centro
        coche_en_grava = distancia_al_centro > der_csv[idx_cercano]


    #Dia4: Para calcular en vez de por velocidad por avance y teniendo en cuenta el final del circuito
    avanzado = self.idx_real - idx_anterior_tick
    haTerminado = False


    if avanzado < -100: # Si ha avanzado más de 100 puntos, es que ha dado la vuelta al circuito, así que sumamos la longitud del circuito para que el avance sea positivo
            avanzado += self.track_data_len
            haTerminado = True # Si ha dado la vuelta al circuito, ha terminado la vuelta

    elif avanzado > 100: # Si ha retrocedido más de 100 puntos, es que ha dado la vuelta al circuito en sentido contrario, así que restamos la longitud del circuito para que el avance sea negativo
            avanzado -= self.track_data_len
            haTerminado = False


    metros_por_punto = self.track_length / self.track_data_len
    avanzado_metros = avanzado * metros_por_punto


    SPEED_REWARD_FACTOR = 0.02 #Factor para evitar que el coche se quede parado



    # 1. ¿Ha chocado o se ha salido? (Castigo máximo)
    if coche_en_grava:
        reward = -500.0
        terminated = True

    # 2. ¿Ha cruzado la meta de forma segura? (Premio máximo)
    elif haTerminado:
        reward = 1000.0
        terminated = True


    elif self.state['speed']/340 < SPEED_REWARD_FACTOR: # Si va muy lento, le damos un pequeño castigo para que no se quede parado
        reward = -5.0
        terminated = False


    # 3. Sigue en pista conduciendo (Premio por ir rápido)
    else:
        # Le damos puntos por la velocidad, pero le restamos 0.1 por cada tick
        # que pasa para que "tenga prisa" en terminar la vuelta
        #reward = velocidad_ms - 0.1
        
        #DIA4: Cambiamos la recompensa para que no dependa de la velocidad, sino de la distancia que avanza, asi el coche no se vuelve loco intentando ir a toda velocidad aunque se salga, 
        # ahora lo importante es avanzar lo máximo posible sin salirse, y para eso le damos puntos por la distancia que avanza
        #---COMENTARIOS DE ARRIBA COMO UN DIARIO, NO RELEVANTE AHORA---


        reward = avanzado_metros * 1 - 0.1 # Le damos puntos por la distancia que avanza, pero le restamos 0.1 por cada tick que pasa para que "tenga prisa" en terminar la vuelta
        terminated = False

    return reward, terminated, avanzado_metros
  
  






  #===========================================================
  #                 CALCULO FISICAS VELOCIDAD
  #===========================================================

  def calcular_velocidad_fisicas_basica(self, pedal):
      #Añadimos friccion por fisicas
    friccion = self.state['speed'] * 0.02

    #Si esta frenando frena mas que si esta acelerando
    if pedal < 0:
      pedal *= 20
    else:
      pedal *= 10

    # 3. Calculamos la nueva velocidad teórica
    nueva_velocidad = self.state['speed'] + pedal - friccion

    # 4. Limitamos entre 0 y 340 km/h (evita que vaya marcha atrás)
    return float(np.clip(nueva_velocidad, 0.0, 340.0))
  






  def calcular_velocidad_fisicas(self, pedal, dt):

    v_ms = self.state['speed'] / 3.6  # Convertimos km/h a m/s
    
    # Parámetros físicos locales (Mantenemos los mismos que en posicion_coche)
    masa = 798.0            
    gravedad = 9.81         
    constante_aero = 2.7    # Usamos tu constante_aero (engloba densidad, sustentación y área)
    
    # Parámetros exclusivos de la velocidad
    rho = 1.225             # Densidad del aire (kg/m^3) para el drag
    CdA = 1.5               # Coeficiente de drag * Área frontal
    Crr = 0.015             # Coeficiente de resistencia a la rodadura
    potencia_max = 750000.0 # Potencia del motor en Watts (aprox 1000 CV)

    # 1. FUERZAS DE RESISTENCIA (Drag aerodinámico y rodadura)
    fuerza_drag = 0.5 * rho * CdA * (v_ms ** 2)
    fuerza_rodadura = Crr * masa * gravedad
    
    # La resistencia solo aplica si el coche se está moviendo
    fuerza_resistencia = fuerza_drag + fuerza_rodadura if v_ms > 0.1 else 0.0

    # 2. FUERZA LONGITUDINAL (Acción del acelerador / freno)
    fuerza_longitudinal = 0.0
    
    if pedal > 0:
        # ACELERANDO
        if v_ms < 15.0: 
            # Límite por tracción mecánica a baja velocidad
            fuerza_motor_max = masa * gravedad * 1.5  
        else:
            # Límite por potencia del motor a alta velocidad: F = P / v
            fuerza_motor_max = potencia_max / v_ms
        
        fuerza_longitudinal = pedal * fuerza_motor_max
        
    elif pedal < 0:
        # FRENANDO
        # El Downforce aplasta el coche y permite frenar mucho más fuerte a alta velocidad
        fuerza_downforce = constante_aero * (v_ms ** 2)
        fuerza_normal = (masa * gravedad) + fuerza_downforce
        
        mu_freno = 1.8 # Coeficiente de fricción extrema
        fuerza_freno_max = mu_freno * fuerza_normal
        
        fuerza_longitudinal = pedal * fuerza_freno_max

    # 3. SEGUNDA LEY DE NEWTON (F_neta = m * a)
    if v_ms > 0.1:
        fuerza_neta = fuerza_longitudinal - fuerza_resistencia
    else:
        # Si está parado, la aerodinámica no lo empuja hacia atrás
        fuerza_neta = max(0.0, fuerza_longitudinal)

    aceleracion = fuerza_neta / masa
    
    # 4. CINEMÁTICA
    nuevo_v_ms = v_ms + (aceleracion * dt)
    nuevo_v_ms = max(0.0, nuevo_v_ms) # Evitar ir marcha atrás
    
    # Devolver en km/h limitando a la velocidad punta teórica (360 km/h)
    return float(np.clip(nuevo_v_ms * 3.6, 0.0, 360.0))
  




  def calcular_velocidad_fisicas_Supervisado(self, pedal, dt):
      #Vamos a usar el modelo fisicas creado a partir de 65km/h
    lim_velo_modelo = 65

    

    if self.state['speed'] < lim_velo_modelo:
        friccion = self.state['speed'] * 0.02
        if pedal < 0:
            nueva_velocidad = self.state['speed'] - abs(pedal) * 20 - friccion
        else:
            nueva_velocidad = self.state['speed'] + pedal * 10 - friccion
    else:
        # Si va a una velocidad donde el modelo de físicas es fiable, usamos el modelo para predecir la delta velocidad
        if pedal < 0:
            freno = abs(pedal) * 1
            aceleracion = 0
        else:
            freno = 0
            aceleracion = pedal * 100

        prediccion_delta_velocidad = self.modelo_fisicas.predict([[self.state['speed'], aceleracion, freno]])[0]
        nueva_velocidad = self.state['speed'] + prediccion_delta_velocidad

    
    return float(np.clip(nueva_velocidad, 0.0, 340.0))



    
  
  #===========================================================
  #                 CALCULO FISICAS POSICION
  #===========================================================

  def calcular_posicion_coche_basico(self, giro):
    #tener en cuenta el angulo coche porque eso hace que avance mas x o y
    #dt tiempo de cada tick
    dt = self.dt
    velocidad_ms = self.state['speed'] / 3.6  # Convertir km/h a m/s

    #trabajamos con radianes para mejor funcionamiento numpy
    #El giro que le damos al coche no es un giro instantaneo, sino que el coche va girando poco a poco, por eso multiplicamos el giro por un factor para que no gire demasiado rapido
    max_giro_rad_por_tick = 0.15
    self.state['angulo'] += giro * max_giro_rad_por_tick

    self.state['car_x_position'] += velocidad_ms * np.cos(self.state['angulo']) * dt
    self.state['car_y_position'] += velocidad_ms * np.sin(self.state['angulo']) * dt
  






  def calcular_posicion_coche(self, giro):
      
    #Variables fisicas del coche
    masa = 798.0
    lon_coc = 3.6 #longitud entre ejes coche
    gravedad = 9.81
    constante_aero = 2.7 #densidad aire nivel mar(1.225) * coeficiente de sustentación(3) * area frontal(1.5)
    velocidad_ms = self.state['speed'] / 3.6  # Convertir km/h a m/s
    mu = 1.6 # Coeficiente de fricción estática de los neumáticos de F1
    lim_giro_mecanico_rad = np.radians(18) # El tope físico de la dirección del F1 (unos 18º)



    #Calcular fuerza adherencia maxima, teniendo en cuenta downforce y centipreta
    fuerza_adh_max = (masa * gravedad) + (constante_aero * velocidad_ms**2)

    #Aceleracion lateral máxima que pueden soportar los neumáticos sin derrapar
    a_lat_max = mu * fuerza_adh_max / masa

    #Tasa de giro máxima que pueden soportar los neumáticos sin derrapar (en rad/s)
    w_max = a_lat_max / max(velocidad_ms, 0.1) # Evitamos división por cero a muy baja velocidad

    #Ángulo de giro máximo permitido a esta velocidad (en radianes)
    angulo_max_giro = np.arctan((w_max * lon_coc) / max(velocidad_ms, 0.1)) # Despejamos el ángulo de la fórmula de la bicicleta: w = (v / L) * tan(angulo)

    lim_giro_real = min(angulo_max_giro, lim_giro_mecanico_rad) # El coche no puede girar más del tope mecánico de la dirección, ni más de lo que la física permite antes de derrapar.
  
    
    angulo_ruedas = giro * lim_giro_real # La acción del agente 'giro' va de -1 a 1. La multiplicamos por el límite real calculado en ese instante.

    w_real = (velocidad_ms / lon_coc) * np.tan(angulo_ruedas) # Tasa de giro real basada en el ángulo de las ruedas

    # Actualizamos el estado del coche
    self.state['angulo'] += w_real * self.dt
    self.state['car_x_position'] += velocidad_ms * np.cos(self.state['angulo']) * self.dt
    self.state['car_y_position'] += velocidad_ms * np.sin(self.state['angulo']) * self.dt





















def make_env():
    # Función auxiliar para crear instancias del entorno
    def _init():
        track_file_path = 'D://Aplicaciones//TFG2//Circuitos//Monza.csv'
        circuito = pd.read_csv(track_file_path)
        env = F1Env(circuito)
        return Monitor(env)
    return _init

if __name__ == '__main__':
    # Si tienes un procesador de 8 núcleos, puedes poner 4 u 8 entornos
    num_cpu = 3 
    env = SubprocVecEnv([make_env() for i in range(num_cpu)])

    model = PPO('MultiInputPolicy', env, verbose=1, tensorboard_log="./Training/logs/")
    model.learn(total_timesteps=5000000)

    shower_path = "./Training/SavedModels/showerPPO/"

    # GUARDAR MODELO
    print("Guardando modelo...")
    model.save(shower_path + "PPO_F1_5M_V8")






























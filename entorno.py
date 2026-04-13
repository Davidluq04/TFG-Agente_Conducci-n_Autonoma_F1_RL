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

import joblib  # Librería para guardar modelos de Machine Learning

class F1Env(Env):
  def __init__(self, track_file_path):

    self.track_data = track_file_path
    self.track_data_len = len(self.track_data)

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
    self.screen = None


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
    self.state['speed'] = float(np.clip(nueva_velocidad, 0.0, 340.0))
    '''

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

    
    self.state['speed'] = float(np.clip(nueva_velocidad, 0.0, 340.0))

    
    #---------------------------CAMBIAMOS POSICION COCHE------------------------

    #tener en cuenta el angulo coche porque eso hace que avance mas x o y
    #dt tiempo de cada tick
    dt = 0.1
    velocidad_ms = self.state['speed'] / 3.6  # Convertir km/h a m/s

    #trabajamos con radianes para mejor funcionamiento numpy
    #El giro que le damos al coche no es un giro instantaneo, sino que el coche va girando poco a poco, por eso multiplicamos el giro por un factor para que no gire demasiado rapido
    max_giro_rad_por_tick = 0.15
    self.state['angulo'] += giro * max_giro_rad_por_tick

    self.state['car_x_position'] += velocidad_ms * np.cos(self.state['angulo']) * dt
    self.state['car_y_position'] += velocidad_ms * np.sin(self.state['angulo']) * dt
    '''



    #Vamos a aplicar modelo cinematica bicicleta para que el coche gire de forma mas realista
    #formula w = v/L * tan(giro)

    #tener en cuenta el angulo coche porque eso hace que avance mas x o y
    #dt tiempo de cada tick
    dt = 0.1
    velocidad_ms = self.state['speed'] / 3.6  # Convertir km/h a m/s

    lon_coc = 3.6
    lim_giro_rue = np.radians(40) # Las ruedas no pueden girar mas de 40 grados, asi que limitamos el giro a ese valor

    angulo_ruedas = giro * lim_giro_rue

    if(velocidad_ms > 0.5):
        w = velocidad_ms / lon_coc * np.tan(angulo_ruedas)
        self.state['angulo'] += w * dt
    else:
        pass # Si el coche va muy lento, no gira para evitar que se vuelva loco

    self.state['car_x_position'] += velocidad_ms * np.cos(self.state['angulo']) * dt
    self.state['car_y_position'] += velocidad_ms * np.sin(self.state['angulo']) * dt

    '''
    #-------------------------FUNCION DE RECOMPENSA--------------------------
    '''
    La funcion de recompensa la he cambiado del dia 2 al 4, en el dia 2 calculaba el idx en base
    a la posicion en porcentaje y eso podia hacer que variase y no fuese exacto, por eso en el dia 4
    he puesto lo de abajo, para ello, he añadido en state un idx_anterior_tick para ir calculando el idx
    constantemente. Tiempo despues He cambiado ahora lo inicio fin para hacer que cuando llegue
    al final del circuito pueda volver a ver los puntos del inicio, para que no se quede sin puntos cercanos
    y el coche no se vuelva loco, ademas de que asi puede hacer varias vueltas sin problemas. 
    '''


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

    #---------------------CALCULAR POSITION, INDICE, DELTA_ANGULO----------------------------

    
    tiempo_vision = 1

    metro_delante = max(10, velocidad_ms * tiempo_vision) # Distancia que recorre en 1.5 segundo a la velocidad actual, para anticiparse a las curvas
    
    puntos_delante = int(metro_delante / metros_por_punto) # Convertimos esa distancia a puntos del CSV

    idx_futuro = (self.idx_real + puntos_delante) % self.track_data_len #punto en el futuro para anticiparse a las curvas, es como los radares pero para curva
    
    x_delante = self.x[idx_futuro]
    y_delante = self.y[idx_futuro]
    x_actual = self.x[self.idx_real]
    y_actual = self.y[self.idx_real]

    angulo_relativo = np.arctan2(y_delante - y_actual, x_delante - x_actual)
    delta_angulo = (angulo_relativo - self.state['angulo'] + np.pi) % (2 * np.pi) - np.pi # Normalizamos el delta de ángulo a un rango de -pi a pi para que la IA lo entienda mejor


    
    por_avanzado = (avanzado/self.track_data_len) * 100
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

  import matplotlib.pyplot as plt

  def render(self):
    
    if self.screen is None:
        pygame.init()

        ancho = 800
        largo = 800
        self. centro_ancho = ancho // 2
        self. centro_largo = largo // 2

        self.escala = 10

        self.fuente = pygame.font.SysFont('Arial', 24, bold=True)

        self.screen = pygame.display.set_mode((ancho, largo))

        coche_img = pygame.image.load('D://Aplicaciones//TFG2//Coche//coche.png') 
        self.imagen_escalada = pygame.transform.scale(coche_img, (50, 20))
        self.imagen_escalada = self.imagen_escalada.convert_alpha()  # Convertir para mejorar el rendimiento y mantener la transparencia

        pygame.display.set_caption('Simulador F1 RL')


    #----PINTAR CIRCUITO----

    self.coche_x = self.state['car_x_position']
    self.coche_y = self.state['car_y_position']
    
    

    puntos_izq_pantalla = []
    puntos_der_pantalla = []

    def pasar_a_pixel(x_m, y_m):
        x_pantalla = int(self.centro_ancho + (x_m - self.coche_x) * self.escala)
        y_pantalla = int(self.centro_largo + (self.coche_y - y_m) * self.escala) # Invertimos Y porque en Pygame el 0 está arriba
        return (x_pantalla, y_pantalla)


    for x_izq, y_izq in zip(self.pared_izq_x, self.pared_izq_y):
        puntos_izq_pantalla.append(pasar_a_pixel(x_izq, y_izq))

    for x_der, y_der in zip(self.pared_der_x, self.pared_der_y):
        puntos_der_pantalla.append(pasar_a_pixel(x_der, y_der))


    for e in pygame.event.get():
       e: pygame.event
       if e.type == QUIT:
           self.close()

    self.screen.fill((34, 134, 34))  # Fondo verde para el césped

    pygame.draw.lines(self.screen, (0, 0, 0), True, puntos_izq_pantalla, 2)
    pygame.draw.lines(self.screen, (0, 0, 0), True, puntos_der_pantalla, 2)

    #----PINTAR COCHE----    

    

    imagen_rotada = pygame.transform.rotate(self.imagen_escalada, np.degrees(self.state['angulo']))

    car_pos = imagen_rotada.get_rect() 
    car_pos.center = (self.centro_ancho, self.centro_largo)  # El coche siempre en el centro de la pantalla

    self.screen.blit(imagen_rotada, car_pos.topleft)  # Dibujar el coche en la pantalla


    #----TELEMETRIA----
    velocidad_texto = int(self.state['speed'])
    texto = self.fuente.render(f"Velocidad: {velocidad_texto} km/h", True, (255, 255, 255))  # Velocidad en porcentaje
    self.screen.blit(texto, (20, 20))  # Mostrar la velocidad en la esquina superior izquierda

    #Aceleracion o frenada

    pedal_pres = self.state.get('pedal', 0)

    pedal_pres = max(-1.0, min(1.0, float(pedal_pres)))

    barra_x = 20
    barra_y = 60
    barra_ancho = 200
    barra_alto = 20
    centro_barra = barra_x + (barra_ancho // 2)
    
    pygame.draw.rect(self.screen, (255, 255, 255), (barra_x, barra_y, barra_ancho, barra_alto), 2)
    pygame.draw.line(self.screen, (255, 255, 255), (centro_barra, barra_y), (centro_barra, barra_y + barra_alto), 2)

    relleno_ancho = int(abs(pedal_pres) * (barra_ancho / 2))

    if pedal_pres > 0.05:
        pygame.draw.rect(self.screen, (0, 255, 0), (centro_barra, barra_y, relleno_ancho, barra_alto))
    elif pedal_pres < -0.05:
        pygame.draw.rect(self.screen, (255, 0, 0), (centro_barra - relleno_ancho, barra_y, relleno_ancho, barra_alto))
        

    #pygame.draw.circle(self.screen, (255, 0, 0), car_pos, 5)  # Coche representado como un círculo rojo

    pygame.time.delay(40)  # Pequeña pausa para controlar la velocidad de renderizado
    pygame.display.flip()

    '''

      # Si es la primera vez que llamamos a render, creamos la ventana y el circuito
      if not hasattr(self, 'fig'):
          plt.ion() # Activar modo interactivo
          self.fig, self.ax = plt.subplots(figsize=(8, 8))
          self.fig.canvas.manager.set_window_title('Simulador F1 RL')
          
          # Dibujamos la pista (usamos la línea central que tienes en el CSV)
          # Le ponemos un grosor grande para simular el asfalto
          self.ax.plot(self.track_data['x_m'], self.track_data['y_m'], color='gray', linewidth=20, alpha=0.5)
          # Línea central punteada
          self.ax.plot(self.track_data['x_m'], self.track_data['y_m'], color='white', linestyle='--', linewidth=1)
          
          # Creamos el marcador del coche (un punto rojo)
          self.car_plot, = self.ax.plot([], [], 'ro', markersize=8, label='Coche')
          self.ax.set_aspect('equal') # Para que las curvas no se deformen
          self.ax.legend()

      # --- En cada frame, actualizamos la posición ---
      
      # 1. Mover el punto rojo a la nueva X e Y
      self.car_plot.set_data([self.state['car_x_position']], [self.state['car_y_position']])
      
      # 2. Mover la cámara para que siga al coche (ventana de 100x100 metros)
      margen = 100
      self.ax.set_xlim(self.state['car_x_position'] - margen, self.state['car_x_position'] + margen)
      self.ax.set_ylim(self.state['car_y_position'] - margen, self.state['car_y_position'] + margen)
      
      # 3. Refrescar la pantalla
      self.fig.canvas.draw()
      self.fig.canvas.flush_events()
      plt.pause(0.001) # Pequeña pausa para que a la pantalla le dé tiempo a pintarse
    '''

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

#CREACION ENTORNO Y PRUEBA DE QUE FUNCIONA
'''
track_file_path = 'D://Aplicaciones//TFG2//Circuitos//Monza.csv'
circuito = pd.read_csv(track_file_path)
print(circuito.columns.tolist())
env = F1Env(circuito)

episodes = 5
for episode in range(episodes):
    obs, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        env.render()
        action = env.action_space.sample()  # Acción aleatoria
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

    print(f'Episode {episode + 1}: Total Reward: {total_reward}')

'''
'''

#CREACION DEL AGENTE
track_file_path = 'D://Aplicaciones//TFG2//Circuitos//Monza.csv'
circuito = pd.read_csv(track_file_path)
env = F1Env(circuito)



log_path = "./Training/logs/"
model = PPO('MultiInputPolicy', env, verbose=1, tensorboard_log=log_path)

#ENTRENAMIENTO DEL AGENTE
print("Empezando entrenamiento...") 
model.learn(total_timesteps=5000000)

shower_path = "./Training/SavedModels/showerPPO/"

#GUARDAR MODELO
print("Guardando modelo...")
model.save(shower_path + "PPO_F1_5M_V4")

'''

# Ruta al modelo guardado
model_path = "./Training/SavedModels/showerPPO/PPO_F1_5M_V5"

#CREACION DEL AGENTE
track_file_path = 'D://Aplicaciones//TFG2//Circuitos//YasMarina.csv'
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

'''

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
    model.save(shower_path + "PPO_F1_5M_V5")
'''
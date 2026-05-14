import pygame
from pygame.locals import QUIT
import numpy as np

class F1Renderer:
    def __init__(self, pared_izq_x, pared_izq_y, pared_der_x, pared_der_y):
        pygame.init()

        ancho = 800
        largo = 800
        self. centro_ancho = ancho // 2
        self. centro_largo = largo // 2

        self.pared_izq_x = pared_izq_x
        self.pared_izq_y = pared_izq_y
        self.pared_der_x = pared_der_x
        self.pared_der_y = pared_der_y

        self.escala = 10

        self.fuente = pygame.font.SysFont('Arial', 24, bold=True)

        self.screen = pygame.display.set_mode((ancho, largo))

        coche_img = pygame.image.load('D://Aplicaciones//TFG2//Coche//coche.png') 
        self.imagen_escalada = pygame.transform.scale(coche_img, (50, 20))
        self.imagen_escalada = self.imagen_escalada.convert_alpha()  # Convertir para mejorar el rendimiento y mantener la transparencia

        pygame.display.set_caption('Simulador F1 RL')

    def render(self, state):
        #----PINTAR CIRCUITO----

        self.coche_x = state['car_x_position']
        self.coche_y = state['car_y_position']
        
        

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

        

        imagen_rotada = pygame.transform.rotate(self.imagen_escalada, np.degrees(state['angulo']))

        car_pos = imagen_rotada.get_rect() 
        car_pos.center = (self.centro_ancho, self.centro_largo)  # El coche siempre en el centro de la pantalla

        self.screen.blit(imagen_rotada, car_pos.topleft)  # Dibujar el coche en la pantalla


        #----TELEMETRIA----
        velocidad_texto = int(state['speed'])
        texto = self.fuente.render(f"Velocidad: {velocidad_texto} km/h", True, (255, 255, 255))  # Velocidad en porcentaje
        self.screen.blit(texto, (20, 20))  # Mostrar la velocidad en la esquina superior izquierda

        #Aceleracion o frenada

        pedal_pres = state.get('pedal', 0)

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

        pygame.time.delay(40)
        pygame.display.flip()

    def close(self):
        pygame.quit()
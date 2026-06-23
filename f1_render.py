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
        self.fuente_btn = pygame.font.SysFont('Arial', 16, bold=True) # Fuente para los botones

        self.screen = pygame.display.set_mode((ancho, largo))

        coche_img = pygame.image.load('./Coche/coche.png') 
        self.imagen_escalada = pygame.transform.scale(coche_img, (50, 20))
        self.imagen_escalada = self.imagen_escalada.convert_alpha()  # Convertir para mejorar el rendimiento y mantener la transparencia

        pygame.display.set_caption('Simulador F1 RL')

        #botones
        self.paused = False
        self.btn_pause = pygame.Rect(ancho - 160, 20, 140, 40)
        self.btn_restart = pygame.Rect(ancho - 160, 70, 140, 40)
        self.btn_quit = pygame.Rect(ancho - 160, 120, 140, 40)

    def render(self, state):

        comando = "CONTINUE" # Comando por defecto

        
        for e in pygame.event.get():
            if e.type == QUIT:
                comando = "QUIT"
            elif e.type == pygame.MOUSEBUTTONDOWN:
                if e.button == 1:  # Clic izquierdo del ratón
                    # Comprobamos si el clic choca con alguno de los botones
                    if self.btn_pause.collidepoint(e.pos):
                        self.paused = not self.paused # Alternamos entre pausa/play
                    elif self.btn_restart.collidepoint(e.pos):
                        comando = "RESTART"
                        self.paused = False # Quitamos la pausa si reiniciamos
                    elif self.btn_quit.collidepoint(e.pos):
                        comando = "QUIT"

        # Si estamos en pausa, sobreescribimos el comando
        if self.paused:
            comando = "PAUSE"

        #PINTAR CIRCUIT

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

        #PINTAR COCHE 

        

        imagen_rotada = pygame.transform.rotate(self.imagen_escalada, np.degrees(state['angulo']))

        car_pos = imagen_rotada.get_rect() 
        car_pos.center = (self.centro_ancho, self.centro_largo)  # El coche siempre en el centro de la pantalla

        self.screen.blit(imagen_rotada, car_pos.topleft)  # Dibujar el coche en la pantalla


        #TELEMETRIA
        velocidad_texto = int(state['speed'])
        texto = self.fuente.render(f"Velocidad: {velocidad_texto} km/h", True, (255, 255, 255))  # Velocidad en porcentaje
        self.screen.blit(texto, (20, 20))  # Mostrar la velocidad

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

        #BOTONES
        if not self.paused:
            color_pausa = (220, 160, 0)  # Amarillo para "PAUSAR"
            texto_pausa = "PAUSAR"
        else:
            color_pausa = (0, 180, 0)  # Verde para "REANUDAR"
            texto_pausa = "REANUDAR"

        pygame.draw.rect(self.screen, color_pausa, self.btn_pause, border_radius=8)
        text_surf_p = self.fuente_btn.render(texto_pausa, True, (255, 255, 255))
        self.screen.blit(text_surf_p, text_surf_p.get_rect(center=self.btn_pause.center))

        # Reiniciar
        pygame.draw.rect(self.screen, (0, 120, 215), self.btn_restart, border_radius=8)
        text_surf_r = self.fuente_btn.render("REINICIAR", True, (255, 255, 255))
        self.screen.blit(text_surf_r, text_surf_r.get_rect(center=self.btn_restart.center))

        # Finalizar
        pygame.draw.rect(self.screen, (225, 6, 0), self.btn_quit, border_radius=8)
        text_surf_q = self.fuente_btn.render("FINALIZAR", True, (255, 255, 255))
        self.screen.blit(text_surf_q, text_surf_q.get_rect(center=self.btn_quit.center))

        pygame.time.delay(40)
        pygame.display.flip()

        return comando

    def close(self):
        pygame.quit()
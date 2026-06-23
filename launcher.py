import customtkinter as ctk
import threading
from main import testear_modelo 


ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue") 

class F1Launcher(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("F1 IA Simulator - Setup")
        self.geometry("550x700")
        self.resizable(False, False)

        self.configuraciones = {
            "Básicas": {
                "V1 (Avance y Supervivencia)": {
                    "circuitos": {
                        "Monza": "./Circuitos/Monza.csv",
                        "Montreal": "./Circuitos/Montreal.csv",
                        "Melbourne": "./Circuitos/Melbourne.csv",
                        "Silverstone": "./Circuitos/Silverstone.csv",
                        "Sochi": "./Circuitos/Sochi.csv",
                        "Yas Marina": "./Circuitos/YasMarina.csv"
                    },
                    "modelos": {
                        "Modelo Básico V1": "./Training/Modelos_Finales/PPO_F1_1M_V5.zip"
                    }
                }
            },
            "Avanzadas": {
                "V1 (Avance y Supervivencia)": {
                    "circuitos": {
                        "Monza": "./Circuitos/Monza.csv",
                        "Montreal": "./Circuitos/Montreal.csv",
                        "Sochi": "./Circuitos/Sochi.csv",
                    },
                    "modelos": {
                        "Modelo Avanzado (10M)": "./Training/Modelos_Finales/PPO_F1_10M_V1.zip"
                    }
                },
                "V2 (Velocidad Ideal y Trazada)": {
                    "circuitos": {
                        "Monza RL (V2)": "./Datasets/FuncionRecompensa/Monza_RL.csv"
                    },
                    "modelos": {
                        "ModeloV2 Vuelta Rápida": "./Training/Modelos_Finales/best_model_15C11.zip",
                        "ModeloV2 Sin variar Hiperparámetros": "./Training/Modelos_Finales/best_model_15C6.zip" 
                    }
                }
            },
            "Supervisadas (Machine Learning)": {
                "V1 (Avance y Supervivencia)": {
                    "circuitos": {
                        "Monza": "./Circuitos/Monza.csv",
                        "Montreal": "./Circuitos/Montreal.csv",
                        "Melbourne": "./Circuitos/Melbourne.csv",
                        "Silverstone": "./Circuitos/Silverstone.csv",
                        "Sochi": "./Circuitos/Sochi.csv",
                        "Yas Marina": "./Circuitos/YasMarina.csv"
                    },
                    "modelos": {
                        "Modelo Supervisado": "./Training/Modelos_Finales/best_model_arbol.zip",
                        "Modelo Supervisado Fallido": "./Training/Modelos_Finales/PPO_F1_5M_V13.zip"
                    }
                }
            }
        }


        #CABECERA
        self.header_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.header_frame.pack(pady=(30, 15), fill="x")

        self.label_titulo = ctk.CTkLabel(self.header_frame, text="F1 RL SIMULATOR", font=ctk.CTkFont(size=28, weight="bold"))
        self.label_titulo.pack()
        
        self.label_subtitulo = ctk.CTkLabel(self.header_frame, text="Panel de Configuración del Entorno", font=ctk.CTkFont(size=14, slant="italic"), text_color="gray")
        self.label_subtitulo.pack()

        #TARJETA PRINCIPA
        self.card_frame = ctk.CTkFrame(self, corner_radius=15, fg_color="#2B2B2B")
        self.card_frame.pack(padx=30, pady=10, fill="both", expand=True)

        # Configuraciones de fuente para las labels
        label_font = ctk.CTkFont(size=14, weight="bold")

        #FÍSICAS
        self.label_fisicas = ctk.CTkLabel(self.card_frame, text="Motor de Físicas", font=label_font, text_color="#A0A0A0")
        self.label_fisicas.pack(pady=(20, 5), padx=30, anchor="w")
        
        self.combo_fisicas = ctk.CTkComboBox(self.card_frame, values=list(self.configuraciones.keys()), 
                                             width=400, height=35, command=self.cambio_fisicas, state="readonly")
        self.combo_fisicas.pack(padx=30, pady=(0, 15))

        #RECOMPENSA
        self.label_recompensa = ctk.CTkLabel(self.card_frame, text="Función de Recompensa", font=label_font, text_color="#A0A0A0")
        self.label_recompensa.pack(pady=(10, 5), padx=30, anchor="w")
        
        self.combo_recompensa = ctk.CTkComboBox(self.card_frame, values=[], 
                                                width=400, height=35, command=self.cambio_recompensa, state="readonly")
        self.combo_recompensa.pack(padx=30, pady=(0, 15))

        # SEPARADOR
        self.separador = ctk.CTkFrame(self.card_frame, height=2, fg_color="#3B3B3B")
        self.separador.pack(fill="x", padx=30, pady=10)

        # CIRCUITO
        self.label_circuito = ctk.CTkLabel(self.card_frame, text="Circuito de Pruebas", font=label_font, text_color="#A0A0A0")
        self.label_circuito.pack(pady=(10, 5), padx=30, anchor="w")
        
        self.combo_circuito = ctk.CTkComboBox(self.card_frame, values=[], width=400, height=35, state="readonly")
        self.combo_circuito.pack(padx=30, pady=(0, 15))

        # MODELO
        self.label_modelo = ctk.CTkLabel(self.card_frame, text="Modelo de Red Neuronal", font=label_font, text_color="#A0A0A0")
        self.label_modelo.pack(pady=(10, 5), padx=30, anchor="w")
        
        self.combo_modelo = ctk.CTkComboBox(self.card_frame, values=[], width=400, height=35, state="readonly")
        self.combo_modelo.pack(padx=30, pady=(0, 20))

        # INCIAR
        self.boton_lanzar = ctk.CTkButton(self, text="INICIAR SIMULACIÓN", 
                                          command=self.iniciar_simulacion, 
                                          font=ctk.CTkFont(size=16, weight="bold"),
                                          height=50, corner_radius=10, 
                                          fg_color="#E10600", hover_color="#B30500") 
        self.boton_lanzar.pack(pady=(20, 30), padx=30, fill="x")

        # Arrancamos con los valores por defecto
        self.combo_fisicas.set("Avanzadas")
        self.cambio_fisicas("Avanzadas")





    def cambio_fisicas(self, eleccion_fisica):
        recompensas_soportadas = list(self.configuraciones[eleccion_fisica].keys())
        self.combo_recompensa.configure(values=recompensas_soportadas)
        self.combo_recompensa.set(recompensas_soportadas[0])
        
        if len(recompensas_soportadas) == 1:
            self.combo_recompensa.configure(state="disabled")
        else:
            self.combo_recompensa.configure(state="normal")
            
        self.cambio_recompensa(recompensas_soportadas[0])

    def cambio_recompensa(self, eleccion_recompensa):
        if "V2" in eleccion_recompensa:
            self.combo_fisicas.configure(state="disabled")
        else:
            self.combo_fisicas.configure(state="readonly")

        eleccion_fisica = self.combo_fisicas.get()
        circuitos = self.configuraciones[eleccion_fisica][eleccion_recompensa]["circuitos"]
        modelos = self.configuraciones[eleccion_fisica][eleccion_recompensa]["modelos"]
        
        self.combo_circuito.configure(values=list(circuitos.keys()))
        self.combo_circuito.set(list(circuitos.keys())[0])
        
        self.combo_modelo.configure(values=list(modelos.keys()))
        self.combo_modelo.set(list(modelos.keys())[0])

    def iniciar_simulacion(self):
        fisicas = self.combo_fisicas.get()
        recompensa = self.combo_recompensa.get()
        n_circuito = self.combo_circuito.get()
        n_modelo = self.combo_modelo.get()
        
        ruta_circuito = self.configuraciones[fisicas][recompensa]["circuitos"][n_circuito]
        ruta_modelo = self.configuraciones[fisicas][recompensa]["modelos"][n_modelo]
        
        self.boton_lanzar.configure(state="disabled", text="EJECUTANDO SIMULACIÓN...")

        hilo_simulacion = threading.Thread(
            target=self.correr_entorno, 
            args=(ruta_circuito, ruta_modelo, fisicas, recompensa)
        )
        hilo_simulacion.start()

    def correr_entorno(self, circuito, modelo, fisicas, recompensa):
        try:
            testear_modelo(circuito, modelo, fisicas, recompensa)
        except Exception as e:
            print(f"Error en la simulación: {e}")
        finally:
            self.boton_lanzar.configure(state="normal", text="INICIAR SIMULACIÓN")

if __name__ == "__main__":
    app = F1Launcher()
    app.mainloop()
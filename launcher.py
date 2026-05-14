import customtkinter as ctk
import threading
from main import testear_modelo 

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class F1Launcher(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("F1 RL Simulator Launcher")
        self.geometry("500x550") # <-- AMPLIADO PARA QUE QUEPA EL BOTÓN
        self.resizable(False, False)

        # ---- DICCIONARIO DE DEPENDENCIAS (NUEVA ESTRUCTURA) ----
        # Orden: Físicas -> Recompensa -> Circuitos y Modelos
        self.configuraciones = {
            "Básicas": {
                "V1 (Avance y Supervivencia)": {
                    "circuitos": {
                        "Monza": "D://Aplicaciones//TFG2//Circuitos//Monza.csv",
                        "Montreal": "D://Aplicaciones//TFG2//Circuitos//Montreal.csv",
                        "Melbourne": "D://Aplicaciones//TFG2//Circuitos//Melbourne.csv",
                        "Silverstone": "D://Aplicaciones//TFG2//Circuitos//Silverstone.csv",
                        "Sochi": "D://Aplicaciones//TFG2//Circuitos//Sochi.csv",
                        "Yas Marina": "D://Aplicaciones//TFG2//Circuitos//YasMarina.csv"
                    },
                    "modelos": {
                        "Modelo Básico V1": "./Training/SavedModels/showerPPO/PPO_F1_5M_V1.zip"
                    }
                }
                # Básicas no tiene V2
            },
            "Avanzadas": {
                "V1 (Avance y Supervivencia)": {
                    "circuitos": {
                        "Montreal": "D://Aplicaciones//TFG2//Circuitos//Montreal.csv",
                        "Monza": "D://Aplicaciones//TFG2//Circuitos//Monza.csv",
                        "Sochi": "D://Aplicaciones//TFG2//Circuitos//Sochi.csv",
                    },
                    "modelos": {
                        "Modelo Avanzado (10M)": "./Training/SavedModels/showerPPO/PPO_F1_10M_V1.zip",
                        "Mejor Modelo Avanzado": "./Training/SavedModels/showerPPO/Mejores_Modelos/best_model.zip"
                    }
                },
                "V2 (Velocidad Ideal y Trazada)": {
                    "circuitos": {
                        "Monza RL (V2)": "D://Aplicaciones//TFG2//Datasets//FuncionRecompensa//Monza_RL.csv"
                    },
                    "modelos": {
                        # pon aquí el nombre de tu modelo entrenado con la recompensa V2
                        "Modelo Entrenado en V2": "./Training/SavedModels/showerPPO/PPO_F1_5M_V10.zip" 
                    }
                }
            },
            "Supervisadas (Machine Learning)": {
                "V1 (Avance y Supervivencia)": {
                    "circuitos": {
                        "Yas Marina": "D://Aplicaciones//TFG2//Circuitos//YasMarina.csv"
                    },
                    "modelos": {
                        "Modelo ML Supervisado": "./Training/SavedModels/showerPPO/Modelo_Supervisado.zip"
                    }
                }
                # Si en el futuro tienes V2 para ML, añádelo aquí con la misma estructura que Avanzadas
            }
        }

        # ---- TÍTULO ----
        self.label_titulo = ctk.CTkLabel(self, text="🏎️ Simulador F1 IA", font=ctk.CTkFont(size=24, weight="bold"))
        self.label_titulo.pack(pady=(20, 10))

        # ---- SELECCIÓN DE FÍSICAS ----
        self.label_fisicas = ctk.CTkLabel(self, text="Selecciona el Motor de Físicas:", font=ctk.CTkFont(size=14))
        self.label_fisicas.pack(pady=(10, 0))
        
        opciones_fisicas = list(self.configuraciones.keys())
        self.combo_fisicas = ctk.CTkComboBox(self, values=opciones_fisicas, width=250, command=self.cambio_fisicas)
        self.combo_fisicas.set("Avanzadas")
        self.combo_fisicas.pack(pady=5)

        # ---- SELECCIÓN DE RECOMPENSA ----
        self.label_recompensa = ctk.CTkLabel(self, text="Función de Recompensa:", font=ctk.CTkFont(size=14))
        self.label_recompensa.pack(pady=(20, 0))
        
        # Corregido: El command se pasa directamente dentro del CTkComboBox
        self.combo_recompensa = ctk.CTkComboBox(self, values=[], width=250, command=self.cambio_recompensa)
        self.combo_recompensa.pack(pady=5)

        # ---- SELECCIÓN DE CIRCUITO ----
        self.label_circuito = ctk.CTkLabel(self, text="Selecciona el Circuito:", font=ctk.CTkFont(size=14))
        self.label_circuito.pack(pady=(20, 0))
        
        self.combo_circuito = ctk.CTkComboBox(self, values=[], width=250)
        self.combo_circuito.pack(pady=5)

        # ---- SELECCIÓN DE MODELO ----
        self.label_modelo = ctk.CTkLabel(self, text="Selecciona el Modelo PPO:", font=ctk.CTkFont(size=14))
        self.label_modelo.pack(pady=(20, 0))
        
        self.combo_modelo = ctk.CTkComboBox(self, values=[], width=250)
        self.combo_modelo.pack(pady=5)

        # ---- BOTÓN LANZAR ----
        self.boton_lanzar = ctk.CTkButton(self, text="🚀 INICIAR SIMULACIÓN", 
                                          command=self.iniciar_simulacion, 
                                          font=ctk.CTkFont(size=16, weight="bold"),
                                          height=40, fg_color="#E10600", hover_color="#900000")
        self.boton_lanzar.pack(pady=(40, 20))

        # Arrancamos con los valores por defecto
        self.cambio_fisicas("Avanzadas")

    # ---- EVENTO: AL CAMBIAR LAS FÍSICAS ----
    def cambio_fisicas(self, eleccion_fisica):
        # 1. Miramos qué recompensas soporta esta física (V1 y/o V2)
        recompensas_soportadas = list(self.configuraciones[eleccion_fisica].keys())
        
        # 2. Actualizamos el menú de recompensas
        self.combo_recompensa.configure(values=recompensas_soportadas)
        self.combo_recompensa.set(recompensas_soportadas[0])
        
        # 3. Si solo hay una recompensa (ej: Básicas), lo bloqueamos para que no confunda
        if len(recompensas_soportadas) == 1:
            self.combo_recompensa.configure(state="disabled")
        else:
            self.combo_recompensa.configure(state="normal")
            
        # 4. Actualizamos los circuitos y modelos de esa combinación
        self.cambio_recompensa(recompensas_soportadas[0])

    # ---- EVENTO: AL CAMBIAR LA RECOMPENSA ----
    def cambio_recompensa(self, eleccion_recompensa):
        eleccion_fisica = self.combo_fisicas.get()
        
        # Cogemos circuitos y modelos directamente de la combinación exacta
        circuitos = self.configuraciones[eleccion_fisica][eleccion_recompensa]["circuitos"]
        modelos = self.configuraciones[eleccion_fisica][eleccion_recompensa]["modelos"]
        
        self.combo_circuito.configure(values=list(circuitos.keys()))
        self.combo_circuito.set(list(circuitos.keys())[0])
        
        self.combo_modelo.configure(values=list(modelos.keys()))
        self.combo_modelo.set(list(modelos.keys())[0])

    # ---- EVENTO: AL HACER CLIC EN EL BOTÓN ----
    def iniciar_simulacion(self):
        fisicas = self.combo_fisicas.get()
        recompensa = self.combo_recompensa.get()
        n_circuito = self.combo_circuito.get()
        n_modelo = self.combo_modelo.get()
        
        ruta_circuito = self.configuraciones[fisicas][recompensa]["circuitos"][n_circuito]
        ruta_modelo = self.configuraciones[fisicas][recompensa]["modelos"][n_modelo]

        print(f"Lanzando -> Físicas: {fisicas} | Recompensa: {recompensa[:2]} | Circuito: {n_circuito}")
        
        self.boton_lanzar.configure(state="disabled", text="EJECUTANDO...")

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
            self.boton_lanzar.configure(state="normal", text="🚀 INICIAR SIMULACIÓN")

if __name__ == "__main__":
    app = F1Launcher()
    app.mainloop()
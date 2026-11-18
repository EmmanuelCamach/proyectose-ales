"""
========================================================================================
INTERFAZ GRÁFICA SENCILLA
Sistema de Reconocimiento de Señales por Análisis de Banco de Filtros
Profesor: Fabián Velásquez Clavijo
========================================================================================
"""

import threading
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import configuracion as cfg
from procesamiento import cargar_audio
from reconocimiento import (
    reconocer_ejemplo_pdf, algoritmo_reconocimiento, hay_canto
)


class InterfazSencilla(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Reconocimiento por Banco de Filtros")
        self.geometry("860x700")
        self.configure(bg="#f8f9fa")
        self.escuchando = False
        self.audio_actual = None      # último audio cargado o grabado
        self.nombre_audio = ""
        self.reproduciendo = False
        self.id_fin = None

        # 1. Barra de botones superior
        self._crear_controles()

        # 2. Reproductor del audio cargado o grabado
        self._crear_reproductor()

        # 3. Panel de resultado
        self._crear_panel_resultado()

        # 3. Gráficas (Espectro y Distancias)
        self._crear_graficas()

        # Iniciar mostrando el ejemplo del PDF
        self.probar_ejemplo_pdf()

    def _crear_controles(self):
        frame_btn = tk.Frame(self, bg="#ffffff", bd=1, relief="groove", pady=10)
        frame_btn.pack(fill=tk.X, padx=12, pady=(10, 6))

        tk.Button(
            frame_btn, text="Cargar Audio WAV", bg="#334155", fg="white",
            font=("Segoe UI", 10), padx=14, pady=5, relief="flat", cursor="hand2",
            command=self.cargar_wav
        ).pack(side=tk.LEFT, padx=8)

        tk.Button(
            frame_btn, text=f"Grabar Micrófono ({cfg.DURACION_CAPTURA:.0f}s)", bg="#334155", fg="white",
            font=("Segoe UI", 10), padx=14, pady=5, relief="flat", cursor="hand2",
            command=self.grabar_microfono
        ).pack(side=tk.LEFT, padx=8)

        self.btn_continuo = tk.Button(
            frame_btn, text="Escucha continua", bg="#334155", fg="white",
            font=("Segoe UI", 10), padx=14, pady=5, relief="flat", cursor="hand2",
            command=self.alternar_continuo
        )
        self.btn_continuo.pack(side=tk.LEFT, padx=8)

        tk.Button(
            frame_btn, text="Ejemplo ", bg="#334155", fg="white",
            font=("Segoe UI", 10), padx=14, pady=5, relief="flat", cursor="hand2",
            command=self.probar_ejemplo_pdf
        ).pack(side=tk.LEFT, padx=8)

        self.lbl_estado = tk.Label(frame_btn, text="", bg="#ffffff", fg="#64748b", font=("Segoe UI", 9))
        self.lbl_estado.pack(side=tk.RIGHT, padx=12)

    def _crear_reproductor(self):
        frame_rep = tk.Frame(self, bg="#ffffff", bd=1, relief="groove", pady=6, padx=12)
        frame_rep.pack(fill=tk.X, padx=12, pady=(0, 6))

        self.btn_reproducir = tk.Button(
            frame_rep, text="▶ Reproducir", bg="#15803d", fg="white",
            font=("Segoe UI", 10), padx=14, pady=3, relief="flat", cursor="hand2",
            state=tk.DISABLED, command=self.alternar_reproduccion
        )
        self.btn_reproducir.pack(side=tk.LEFT)

        self.lbl_audio = tk.Label(
            frame_rep, text="Ningún audio cargado", bg="#ffffff", fg="#334155",
            font=("Segoe UI", 9), anchor="w"
        )
        self.lbl_audio.pack(side=tk.LEFT, padx=12, fill=tk.X, expand=True)

    def _actualizar_reproductor(self, audio, nombre):
        """Guarda el audio para poder escucharlo y habilita el botón."""
        self.detener_reproduccion()
        self.audio_actual = np.asarray(audio, dtype=np.float32)
        self.nombre_audio = nombre
        duracion = len(self.audio_actual) / cfg.SAMPLE_RATE
        self.lbl_audio.config(text=f"{nombre}   ({duracion:.1f} s)")
        self.btn_reproducir.config(state=tk.NORMAL)

    def alternar_reproduccion(self):
        if self.reproduciendo:
            self.detener_reproduccion()
            return
        if self.audio_actual is None:
            return
        try:
            import sounddevice as sd
        except ImportError:
            messagebox.showerror("Error", "Falta librería sounddevice: pip install sounddevice")
            return
        if self.escuchando:
            messagebox.showinfo("Reproducir", "Detenga la escucha continua antes de reproducir.")
            return
        try:
            sd.play(self.audio_actual, cfg.SAMPLE_RATE)
        except Exception as e:
            messagebox.showerror("Error al reproducir", f"{e}")
            return
        self.reproduciendo = True
        self.btn_reproducir.config(text="■ Detener", bg="#b91c1c")
        duracion_ms = int(1000 * len(self.audio_actual) / cfg.SAMPLE_RATE) + 200
        self.id_fin = self.after(duracion_ms, self._fin_reproduccion)

    def detener_reproduccion(self):
        if self.reproduciendo:
            try:
                import sounddevice as sd
                sd.stop()
            except Exception:
                pass
        self._fin_reproduccion()

    def _fin_reproduccion(self):
        if self.id_fin is not None:
            self.after_cancel(self.id_fin)
            self.id_fin = None
        self.reproduciendo = False
        self.btn_reproducir.config(text="▶ Reproducir", bg="#15803d")

    def _crear_panel_resultado(self):
        frame_res = tk.Frame(self, bg="#ffffff", bd=1, relief="groove", pady=8, padx=12)
        frame_res.pack(fill=tk.X, padx=12, pady=(0, 6))

        self.lbl_resultado = tk.Label(
            frame_res, text="---", bg="#f1f5f9", fg="#0f172a",
            font=("Segoe UI", 12, "bold"), pady=6, relief="solid", bd=1
        )
        self.lbl_resultado.pack(fill=tk.X, pady=(0, 4))

        self.lbl_metricas = tk.Label(
            frame_res, text="Ex: --- | dA: --- | dB: ---", bg="#ffffff", fg="#334155",
            font=("Consolas", 10)
        )
        self.lbl_metricas.pack()

    def _crear_graficas(self):
        frame_graf = tk.Frame(self, bg="#ffffff", bd=1, relief="groove")
        frame_graf.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 10))

        self.fig = Figure(figsize=(8, 5), dpi=95)
        self.fig.patch.set_facecolor("#ffffff")
        self.ax1 = self.fig.add_subplot(2, 1, 1)
        self.ax2 = self.fig.add_subplot(2, 1, 2)
        self.fig.tight_layout(pad=2.2)

        self.canvas = FigureCanvasTkAgg(self.fig, master=frame_graf)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ------------------------------------------------------------------------
    # MÉTODOS DE ACCIÓN
    # ------------------------------------------------------------------------
    def probar_ejemplo_pdf(self):
        """Ejecuta el ejemplo didáctico exacto de 8 puntos del PDF."""
        self.lbl_estado.config(text="Ejecutando modelo de presustentación ...")
        res = reconocer_ejemplo_pdf()

        # Actualizar texto
        self.lbl_resultado.config(
            text=f"Reconocido: {res['reconocida']}",
            bg="#e2e8f0"
        )
        self.lbl_metricas.config(
            text=f"Ex = [{res['E1']:.2f}, {res['E2']:.2f}, {res['E3']:.2f}]   |   dA = {res['dA']:.2f}   |   dB = {res['dB']:.2f}  (dB < dA => Tipo B)"
        )

        # Graficar
        self.ax1.clear()
        self.ax2.clear()

        # Subplot 1: Stem del espectro discreto
        k = np.arange(8)
        self.ax1.stem(k, np.abs(res["X"]), linefmt="b-", markerfmt="bo", basefmt="k-")
        self.ax1.axvline(2.5, color="gray", linestyle="--", label="Frontera X1 - X2")
        self.ax1.axvline(4.5, color="black", linestyle="--", label="Frontera X2 - X3")
        self.ax1.set_title("Espectro |X(k)| y Subbandas (Ejemplo PDF N=8)", fontsize=10, fontweight="bold")
        self.ax1.set_xlabel("k")
        self.ax1.set_ylabel("Magnitud")
        self.ax1.grid(True, alpha=0.3)
        self.ax1.legend(loc="upper left", fontsize=8)

        # Subplot 2: Barras de distancias
        colores = ["#475569" if res["dA"] < res["dB"] else "#94a3b8",
                   "#475569" if res["dB"] < res["dA"] else "#94a3b8"]
        barras = self.ax2.bar(["dA (Tipo A)", "dB (Tipo B)"], [res["dA"], res["dB"]], color=colores, width=0.4)
        self.ax2.set_ylim(0, 1.2 * max(res["dA"], res["dB"]))
        self.ax2.set_ylabel("Distancia Euclidiana")
        self.ax2.set_title(f"Decisión: Mínima Distancia -> {res['reconocida']}", fontsize=10, fontweight="bold")
        self.ax2.grid(True, alpha=0.3, axis="y")

        for b in barras:
            h = b.get_height()
            self.ax2.text(b.get_x() + b.get_width() / 2.0, h + 1, f"{h:.2f}", ha='center', va='bottom', fontweight='bold')

        self.fig.tight_layout(pad=2.2)
        self.canvas.draw()
        self.lbl_estado.config(text="Ejemplo  verificado.")

    def cargar_wav(self):
        ruta = filedialog.askopenfilename(
            title="Seleccionar archivo WAV",
            filetypes=[("Archivos WAV", "*.wav"), ("Todos los archivos", "*.*")]
        )
        if not ruta:
            return

        try:
            self.lbl_estado.config(text="Procesando audio...")
            audio = cargar_audio(ruta)
            self._actualizar_reproductor(audio, "Audio cargado")
            res = algoritmo_reconocimiento(audio)
            self._mostrar_resultado_audio(res)
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo procesar el archivo:\n{e}")
            self.lbl_estado.config(text="Error al procesar.")

    def grabar_microfono(self):
        self.detener_reproduccion()
        try:
            import sounddevice  # noqa: F401
        except ImportError:
            messagebox.showerror("Error", "Falta librería sounddevice: pip install sounddevice")
            return
        self.lbl_estado.config(text=f"Grabando {cfg.DURACION_CAPTURA:.0f} s por micrófono...")
        threading.Thread(target=self._hilo_rec, daemon=True).start()

    def _capturar_y_reconocer(self):
        """Graba DURACION_CAPTURA segundos, detecta silencio y reconoce."""
        import sounddevice as sd
        fs = cfg.SAMPLE_RATE
        rec = sd.rec(int(cfg.DURACION_CAPTURA * fs), samplerate=fs, channels=1, dtype='float32')
        sd.wait()
        audio = rec[:, 0].astype(float)
        if not hay_canto(audio):
            self.after(0, self._mostrar_silencio)
            return
        res = algoritmo_reconocimiento(audio)
        self.after(0, lambda: self._mostrar_resultado_audio(res))

    def _hilo_rec(self):
        try:
            self._capturar_y_reconocer()
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Error Micrófono", f"{e}"))
            self.after(0, lambda: self.lbl_estado.config(text="Error micrófono."))

    def alternar_continuo(self):
        if self.escuchando:
            self.escuchando = False
            self.btn_continuo.config(text="Escucha continua", bg="#334155")
            self.lbl_estado.config(text="Deteniendo al terminar la captura actual...")
            return
        try:
            import sounddevice  # noqa: F401
        except ImportError:
            messagebox.showerror("Error", "Falta librería sounddevice: pip install sounddevice")
            return
        self.detener_reproduccion()
        self.escuchando = True
        self.btn_continuo.config(text="DETENER escucha", bg="#b91c1c")
        self.lbl_estado.config(text=f"Micrófono activo: reconoce cada {cfg.DURACION_CAPTURA:.0f} s")
        threading.Thread(target=self._hilo_continuo, daemon=True).start()

    def _hilo_continuo(self):
        try:
            while self.escuchando:
                self._capturar_y_reconocer()
            self.after(0, lambda: self.lbl_estado.config(text="Escucha continua detenida."))
        except Exception as e:
            self.escuchando = False
            self.after(0, lambda: self.btn_continuo.config(text="Escucha continua", bg="#334155"))
            self.after(0, lambda: messagebox.showerror("Error Micrófono", f"{e}"))

    def _mostrar_silencio(self):
        self.lbl_resultado.config(text="Sin canto detectado (acerque el parlante o suba el volumen)",
                                  bg="#fef3c7")
        self.lbl_metricas.config(text="Ex: --- | dA: --- | dB: ---")
        self.lbl_estado.config(text="Sin canto." + (" Escuchando..." if self.escuchando else ""))

    def _mostrar_resultado_audio(self, res):
        # Actualizar texto
        self.lbl_resultado.config(
            text=f"Reconocido: {res['reconocida']}",
            bg="#e2e8f0"
        )
        self.lbl_metricas.config(
            text=f"Ex = [{res['Ex'][0]:.4f}, {res['Ex'][1]:.4f}, {res['Ex'][2]:.4f}]   |   dA = {res['dA']:.4f}   |   dB = {res['dB']:.4f}"
        )

        # Graficar
        self.ax1.clear()
        self.ax2.clear()

        # Subplot 1: Espectro y límites de subbandas
        self.ax1.plot(res["frecuencias"], res["magnitud"], color="#334155", lw=1.2)
        for f in cfg.LIMITES_HZ:
            self.ax1.axvline(f, color="gray", linestyle="--", alpha=0.7)
        self.ax1.set_title(f"Espectro y Subbandas: {res['reconocida']}", fontsize=10, fontweight="bold")
        self.ax1.set_xlabel("Frecuencia (Hz)")
        self.ax1.set_ylabel("Magnitud")
        self.ax1.set_xlim(0, 10000)
        self.ax1.grid(True, alpha=0.3)

        # Subplot 2: Distancias
        colores = ["#475569" if res["dA"] < res["dB"] else "#94a3b8",
                   "#475569" if res["dB"] < res["dA"] else "#94a3b8"]
        barras = self.ax2.bar(["dA (Hormiguero)", "dB (Turpial)"], [res["dA"], res["dB"]], color=colores, width=0.4)
        self.ax2.set_ylim(0, 1.2 * max(res["dA"], res["dB"]))
        self.ax2.set_ylabel("Distancia Euclidiana")
        self.ax2.set_title(f"Decisión: Mínima Distancia -> {res['reconocida']}", fontsize=10, fontweight="bold")
        self.ax2.grid(True, alpha=0.3, axis="y")

        for b in barras:
            h = b.get_height()
            self.ax2.text(b.get_x() + b.get_width() / 2.0, h, f"{h:.4f}", ha='center', va='bottom', fontweight='bold')

        self.fig.tight_layout(pad=2.2)
        self.canvas.draw()
        self.lbl_estado.config(text="Escuchando..." if self.escuchando else "Reconocimiento finalizado.")


def lanzar_interfaz():
    app = InterfazSencilla()
    app.mainloop()


if __name__ == "__main__":
    lanzar_interfaz()
"""
========================================================================================
SISTEMA DE RECONOCIMIENTO DE SEÑALES POR ANÁLISIS DE BANCO DE FILTROS
Curso: Procesamiento de Señales e Imágenes / Análisis de Señales
Profesor: Fabián Velásquez Clavijo (2026)

CONDICIONES ESTRICTAS DE SUSTENTACIÓN:
1. Las variables del algoritmo son las mismas variables matemáticas del ejemplo de la
   presustentación: x, xp, xi, G, H, W8, X, X1, X2, X3, E1, E2, E3, Ex, EA, EB, dA, dB.
2. Los vectores promedio de energías (EA, EB) están definidos como CONSTANTES en el código.
3. El algoritmo de reconocimiento NO procesa el banco de grabaciones.
4. Implementa el algoritmo exacto del modelo matemático de la presustentación:
   FFT por submuestreo par/impar, 3 subbandas, energías y distancia euclidiana.
========================================================================================
"""

import sys
import argparse
import numpy as np
import configuracion as cfg
from procesamiento import cargar_audio, preprocesar, seleccionar_segmento, fft_par_impar

# ============================================================================
# VECTORES PROMEDIO DE ENERGÍA (CONSTANTES OBTENIDAS PREVIAMENTE CON entrenamiento.py)
# ============================================================================
# Cantos de eBird / Macaulay Library grabados con el celular (204 grabaciones:
# 109 Hormiguero Gorjinegro, 95 Turpial Amazónico).
# Subbandas: X1 = X(0..127) 0-3000 Hz, X2 = X(128..255) 3000-6000 Hz,
#            X3 = X(256..1024) 6000-24000 Hz.  Energías relativas (suman 1).
# EA: Hormiguero Gorjinegro (Especie A)
EA = np.array([0.02915494, 0.96729073, 0.00355433])
# EB: Turpial Amazónico   (Especie B)
EB = np.array([0.93043191, 0.05695547, 0.01261263])

# Constantes del Ejemplo Didáctico de 8 Puntos (PDF Presustentación)
# Subbandas: X1 = [k=0,1,2], X2 = [k=3,4], X3 = [k=5,6,7]
EA_TOY = np.array([125.0 / 3.0, 3.0, 5.0 / 3.0])  # [41.67, 3.00, 1.67]
EB_TOY = np.array([11.0 / 3.0, 51.0, 7.0 / 3.0])   # [3.67, 51.00, 2.33]


# ============================================================================
# ALGORITMO 1: RECONOCIMIENTO DEL EJEMPLO DIDÁCTICO DE 8 PUNTOS (DEL PDF)
# ============================================================================
def reconocer_ejemplo_pdf(x=None):
    """
    Ejecuta el algoritmo de reconocimiento exacto de la presustentación (PDF)
    para una señal discreta de N = 8 puntos.
    
    Variables idénticas al modelo matemático:
      x   : señal desconocida de 8 puntos
      xp  : muestras pares (x0, x2, x4, x6)
      xi  : muestras impares (x1, x3, x5, x7)
      G   : transformada de 4 puntos de muestras pares
      H   : transformada de 4 puntos de muestras impares
      W8  : factores de rotación twiddle e^{-j*pi*k/4}
      X   : transformada combinada de 8 puntos: X(k) = G(k) + W8^k * H(k)
      X1, X2, X3 : partición en 3 subbandas (tamaños 3, 2, 3)
      E1, E2, E3 : energías promedio por subbanda
      Ex  : vector de energías de la señal: Ex = (E1, E2, E3)
      dA, dB : distancias euclidianas a los vectores promedio EA y EB
    """
    if x is None:
        # Señal desconocida de prueba del PDF (página 6)
        x = np.array([2, -1, 1, -1, 1, -1, 1, -1], dtype=float)
    else:
        x = np.asarray(x, dtype=float)
        if len(x) != 8:
            raise ValueError("La señal del ejemplo de presustentación debe tener exactamente 8 puntos.")

    # 1. Separar muestras pares e impares
    xp = x[0::2]  # (x0, x2, x4, x6)
    xi = x[1::2]  # (x1, x3, x5, x7)

    # 2. Transformadas de 4 puntos (usando submuestreo par/impar como en el PDF)
    # G(k) = [x0+x4, x0-x4] + (-j)^k * [x2+x6, x2-x6]
    # H(k) = [x1+x5, x1-x5] + (-j)^k * [x3+x7, x3-x7]
    G = np.array([
        (xp[0] + xp[2]) + 1 * (xp[1] + xp[3]),
        (xp[0] - xp[2]) - 1j * (xp[1] - xp[3]),
        (xp[0] + xp[2]) - 1 * (xp[1] + xp[3]),
        (xp[0] - xp[2]) + 1j * (xp[1] - xp[3])
    ], dtype=complex)

    H = np.array([
        (xi[0] + xi[2]) + 1 * (xi[1] + xi[3]),
        (xi[0] - xi[2]) - 1j * (xi[1] - xi[3]),
        (xi[0] + xi[2]) - 1 * (xi[1] + xi[3]),
        (xi[0] - xi[2]) + 1j * (xi[1] - xi[3])
    ], dtype=complex)

    # 3. Combinación con el factor W8 = e^{-j*pi/4}:
    # X(k) = G(k) + W8^k * H(k), donde G y H se repiten con periodo 4
    k = np.arange(8)
    W8_k = np.exp(-1j * (np.pi / 4.0) * k)
    G_ext = np.tile(G, 2)
    H_ext = np.tile(H, 2)
    X = G_ext + W8_k * H_ext

    # 4. Partición en 3 subbandas (tamaños 3, 2, 3)
    X1 = X[0:3]  # (X(0), X(1), X(2))
    X2 = X[3:5]  # (X(3), X(4))
    X3 = X[5:8]  # (X(5), X(6), X(7))

    # 5. Cálculo de energías promedio por subbanda
    E1 = (1.0 / 3.0) * np.sum(np.abs(X1) ** 2)
    E2 = (1.0 / 2.0) * np.sum(np.abs(X2) ** 2)
    E3 = (1.0 / 3.0) * np.sum(np.abs(X3) ** 2)
    Ex = np.array([E1, E2, E3])

    # 6. Distancias euclidianas a los vectores promedio constantes (EA_TOY, EB_TOY)
    dA = float(np.sqrt(np.sum((Ex - EA_TOY) ** 2)))
    dB = float(np.sqrt(np.sum((Ex - EB_TOY) ** 2)))

    # 7. Regla de decisión
    reconocida = "Tipo A" if dA < dB else "Tipo B"

    return {
        "x": x, "xp": xp, "xi": xi,
        "G": G, "H": H, "X": X,
        "X1": X1, "X2": X2, "X3": X3,
        "E1": E1, "E2": E2, "E3": E3,
        "Ex": Ex,
        "EA": EA_TOY, "EB": EB_TOY,
        "dA": dA, "dB": dB,
        "reconocida": reconocida
    }


# ============================================================================
# ALGORITMO 2: RECONOCIMIENTO DE AUDIOS DE AVES (PROYECTO COMPLETO)
# ============================================================================
def hay_canto(audio):
    """Energía media (1/N) sum x(n)^2 por encima del umbral de silencio."""
    return float(np.mean(np.asarray(audio, dtype=float) ** 2)) >= cfg.UMBRAL_SILENCIO


def extraer_energias(audio):
    """
    Mismo modelo matemático de la presustentación, aplicado a un audio:
      x          : señal filtrada (Butterworth pasabandas) y recortada a los 3 s de mayor energía
      xp, xi     : muestras pares e impares de cada bloque de N = 2048
      G, H       : FFT de N/2 puntos de xp y xi
      X          : X(k) = G(k) + W^k H(k),  X(k+N/2) = G(k) - W^k H(k)
      X1, X2, X3 : las 3 subbandas del banco de filtros
      E1, E2, E3 : energías promedio  E_i = (1/L_i) sum |X(k)|^2
      Ex         : vector de energías relativas  Ex = (E1, E2, E3) / (E1 + E2 + E3)
    """
    # 1. Preprocesamiento: filtro Butterworth pasabandas y selección de 3 segundos
    x = seleccionar_segmento(preprocesar(audio))
    N = cfg.N_FFT
    if len(x) < N:
        x = np.concatenate([x, np.zeros(N - len(x))])

    # 2. Bloques de N muestras y FFT por submuestreo par/impar
    bloques = x[:(len(x) // N) * N].reshape(-1, N)
    xp = bloques[:, 0::2]                          # muestras pares
    xi = bloques[:, 1::2]                          # muestras impares
    G = fft_par_impar(xp)                          # G(k), N/2 puntos
    H = fft_par_impar(xi)                          # H(k), N/2 puntos
    k = np.arange(N // 2)
    W = np.exp(-1j * 2 * np.pi * k / N)            # W^k = e^{-j 2 pi k / N}
    X = np.concatenate([G + W * H, G - W * H], axis=1)   # X(k), k = 0 .. N-1

    # 3. Partición en 3 subbandas: X1 = X(0..127), X2 = X(128..255), X3 = X(256..1024)
    f0, f1, f2, f3 = cfg.FRONTERAS_SUBBANDAS
    X1 = X[:, f0:f1]
    X2 = X[:, f1:f2]
    X3 = X[:, f2:f3]

    # 4. Energía promedio de cada subbanda (promedio de los bloques): E = (1/L) sum |X(k)|^2
    E1 = float(np.mean((1.0 / (f1 - f0)) * np.sum(np.abs(X1) ** 2, axis=1)))
    E2 = float(np.mean((1.0 / (f2 - f1)) * np.sum(np.abs(X2) ** 2, axis=1)))
    E3 = float(np.mean((1.0 / (f3 - f2)) * np.sum(np.abs(X3) ** 2, axis=1)))

    # 5. Energías relativas: no dependen del volumen ni de la distancia al micrófono
    Et = E1 + E2 + E3
    Ex = np.array([E1, E2, E3]) / Et

    magnitud = np.mean(np.abs(X[:, :N // 2 + 1]), axis=0)
    frecuencias = np.arange(N // 2 + 1) * cfg.DELTA_F
    return {"x": x, "E1": E1, "E2": E2, "E3": E3, "Ex": Ex,
            "magnitud": magnitud, "frecuencias": frecuencias}


def algoritmo_reconocimiento(audio, EA_ref=EA, EB_ref=EB):
    """
    Distancias euclidianas a las constantes EA y EB y regla de mínima distancia:
        dA = ||Ex - EA||,   dB = ||Ex - EB||
    """
    res = extraer_energias(audio)
    Ex = res["Ex"]

    # 6. Distancias euclidianas a los vectores constantes de referencia
    dA = float(np.sqrt(np.sum((Ex - EA_ref) ** 2)))
    dB = float(np.sqrt(np.sum((Ex - EB_ref) ** 2)))

    # 7. Regla de decisión de mínima distancia
    if dA < dB:
        reconocida = "Hormiguero Gorjinegro (Especie A)"
        clave = "HG"
    else:
        reconocida = "Turpial Amazónico (Especie B)"
        clave = "TA"

    res.update({"EA": EA_ref, "EB": EB_ref, "dA": dA, "dB": dB,
                "reconocida": reconocida, "clave": clave})
    return res


# ============================================================================
# FUNCIONES DE IMPRESIÓN Y EJECUCIÓN POR CONSOLA
# ============================================================================
def imprimir_resultado_ejemplo_pdf(res):
    print("=" * 70)
    print("DEMOSTRACIÓN DEL MODELO MATEMÁTICO DEL PDF (PRESUSTENTACIÓN)")
    print("=" * 70)
    print(f"Señal desconocida x = {res['x'].astype(int).tolist()}")
    print(f"Muestras pares   xp = {res['xp'].astype(int).tolist()}")
    print(f"Muestras impares xi = {res['xi'].astype(int).tolist()}")
    print(f"\nTransformada G(k) (pares):\n{np.round(res['G'], 2)}")
    print(f"\nTransformada H(k) (impares):\n{np.round(res['H'], 2)}")
    print(f"\nTransformada combinada X(k):\n{np.round(res['X'], 2)}")
    print(f"\nPartición en 3 subbandas:")
    print(f"  X1 = {np.round(res['X1'], 2)}")
    print(f"  X2 = {np.round(res['X2'], 2)}")
    print(f"  X3 = {np.round(res['X3'], 2)}")
    print(f"\nEnergías promedio de subbandas:")
    print(f"  E1 = {res['E1']:.4f}")
    print(f"  E2 = {res['E2']:.4f}")
    print(f"  E3 = {res['E3']:.4f}")
    print(f"  Vector Ex = {np.round(res['Ex'], 2).tolist()}")
    print(f"\nConstantes de referencia:")
    print(f"  EA (Tipo A) = {np.round(res['EA'], 2).tolist()}")
    print(f"  EB (Tipo B) = {np.round(res['EB'], 2).tolist()}")
    print(f"\nDistancias euclidianas:")
    print(f"  dA = ||Ex - EA|| = {res['dA']:.4f} (Esperado PDF: ~55.66)")
    print(f"  dB = ||Ex - EB|| = {res['dB']:.4f} (Esperado PDF: ~10.43)")
    print(f"\n>>> Decisión: dB < dA => Señal reconocida como: {res['reconocida']} <<<")
    print("=" * 70)


def imprimir_resultado_audio(res, origen="Audio"):
    print("=" * 70)
    print(f"RECONOCIMIENTO DE SEÑAL: {origen}")
    print("=" * 70)
    print(f"Energías: E1 = {res['E1']:.4f}   E2 = {res['E2']:.4f}   E3 = {res['E3']:.4f}")
    print(f"Vector de energías relativas Ex = {np.round(res['Ex'], 4).tolist()}")
    print(f"\nConstantes de referencia:")
    print(f"  EA (Hormiguero Gorjinegro) = {np.round(res['EA'], 4).tolist()}")
    print(f"  EB (Turpial Amazónico)   = {np.round(res['EB'], 4).tolist()}")
    print(f"\nDistancias euclidianas:")
    print(f"  dA = ||Ex - EA|| = {res['dA']:.4f}")
    print(f"  dB = ||Ex - EB|| = {res['dB']:.4f}")
    print(f"\n>>> RESULTADO: {res['reconocida']} <<<")
    print("=" * 70)


# ============================================================================
# CAPTURA POR MICRÓFONO EN TIEMPO REAL
# ============================================================================
def capturar_microfono():
    try:
        import sounddevice as sd
    except ImportError:
        print("[ERROR] Se requiere 'sounddevice' para capturar micrófono: pip install sounddevice")
        return

    import matplotlib.pyplot as plt
    plt.ion()
    fig, ejes = plt.subplots(2, 1, figsize=(9, 6))
    duracion = cfg.DURACION_CAPTURA
    tamano_buffer = int(cfg.SAMPLE_RATE * duracion)

    print("\n[MICRÓFONO ACTIVADO] Captura continua de 6 segundos. Reproduzca el canto...")
    print("Presiona Ctrl+C en la consola o cierra la ventana para detener.")

    try:
        with sd.InputStream(samplerate=cfg.SAMPLE_RATE, channels=1, dtype='float32',
                            blocksize=tamano_buffer) as stream:
            while True:
                audio, desborde = stream.read(tamano_buffer)
                if desborde:
                    print("Aviso: Desbordamiento de buffer, repita la emisión.")
                    continue
                if not hay_canto(audio[:, 0]):
                    print("Sin canto detectado.")
                    continue

                res = algoritmo_reconocimiento(audio[:, 0])
                imprimir_resultado_audio(res, origen="Micrófono en Vivo")

                # Gráfica 1: Espectro y subbandas
                ejes[0].clear()
                ejes[0].plot(res["frecuencias"], res["magnitud"], color="#1f77b4", label="|X(f)|")
                for f in cfg.LIMITES_HZ:
                    ejes[0].axvline(f, color="red", linestyle="--", alpha=0.7)
                ejes[0].set_title(f"Espectro: {res['reconocida']}")
                ejes[0].set_xlabel("Frecuencia (Hz)")
                ejes[0].set_ylabel("Magnitud")
                ejes[0].grid(True, alpha=0.3)

                # Gráfica 2: Distancias euclidianas
                ejes[1].clear()
                colores = ["#2ca02c" if res["dA"] < res["dB"] else "#d62728",
                           "#2ca02c" if res["dB"] < res["dA"] else "#d62728"]
                ejes[1].bar(["Hormiguero (dA)", "Turpial (dB)"], [res["dA"], res["dB"]], color=colores)
                ejes[1].set_ylabel("Distancia Euclidiana")
                ejes[1].set_title("Comparación contra Vectores Constantes (EA, EB)")
                ejes[1].grid(True, alpha=0.3, axis="y")

                fig.tight_layout()
                plt.pause(0.01)
                if not plt.fignum_exists(fig.number):
                    break
    except KeyboardInterrupt:
        print("\nCaptura finalizada.")
    finally:
        plt.close(fig)


# ============================================================================
# PUNTO DE ENTRADA PRINCIPAL
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Sistema de Reconocimiento por Banco de Filtros (Prof. Fabián Velásquez Clavijo)"
    )
    parser.add_argument("--archivo", type=str, help="Ruta de un archivo WAV para reconocimiento.")
    parser.add_argument("--ejemplo-pdf", action="store_true", help="Ejecuta la prueba matemática de 8 puntos del PDF.")
    parser.add_argument("--microfono", action="store_true", help="Activa captura por micrófono en tiempo real.")
    parser.add_argument("--gui", action="store_true", help="Lanza la interfaz gráfica.")

    args = parser.parse_args()

    if args.ejemplo_pdf:
        res = reconocer_ejemplo_pdf()
        imprimir_resultado_ejemplo_pdf(res)
        return

    if args.archivo:
        audio = cargar_audio(args.archivo)
        res = algoritmo_reconocimiento(audio)
        imprimir_resultado_audio(res, origen=args.archivo)
        return

    if args.microfono:
        capturar_microfono()
        return

    # Si se pide --gui o no se pasa ningún argumento, abrir la interfaz gráfica
    try:
        from interfaz import lanzar_interfaz
        lanzar_interfaz()
    except Exception as e:
        print(f"No fue posible abrir la interfaz gráfica ({e}). Ejecutando ejemplo del PDF por consola:")
        res = reconocer_ejemplo_pdf()
        imprimir_resultado_ejemplo_pdf(res)


if __name__ == "__main__":
    main()

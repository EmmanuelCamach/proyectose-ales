"""
Preprocesamiento y FFT, programados con sus formulas:
  - Filtro Butterworth pasabandas: prewarping + transformacion bilineal.
  - Ecuacion en diferencias.
  - Seleccion de los 3 s de mayor energia.
  - FFT por submuestreo par/impar.
"""
import wave
import numpy as np
import configuracion as cfg


# ============================================================================
# LECTURA DE AUDIO
# ============================================================================
def cargar_audio(ruta):
    """Lee un WAV PCM de 16 bits y lo escala a [-1, 1]. Si es estereo se promedia."""
    with wave.open(str(ruta), 'rb') as w:
        fs, canales, ancho = w.getframerate(), w.getnchannels(), w.getsampwidth()
        datos = w.readframes(w.getnframes())
    if fs != cfg.SAMPLE_RATE:
        raise ValueError(f'{ruta}: se requiere fs = {cfg.SAMPLE_RATE} Hz (tiene {fs} Hz).')
    if ancho != 2:
        raise ValueError(f'{ruta}: se requiere WAV PCM de 16 bits.')
    x = np.frombuffer(datos, dtype='<i2').astype(float) / 32768.0
    if canales > 1:
        x = x.reshape(-1, canales).mean(axis=1)
    return x


# ============================================================================
# FILTRO BUTTERWORTH PASABANDAS (PREWARPING + BILINEAL)
# ============================================================================
def prewarping(f, fs):
    """w = (2/T) tan(pi f / fs) = 2 fs tan(pi f / fs)"""
    return 2 * fs * np.tan(np.pi * f / fs)


def butterworth_pasabandas(f1, f2, fs):
    """
    Prototipo de orden 2:      H(s) = 1 / (s^2 + sqrt(2) s + 1)
    Pasabajas -> pasabandas:   s <- (s^2 + w0^2) / (BW s),  w0^2 = w1 w2,  BW = w2 - w1

                         BW^2 s^2
    H(s) = ------------------------------------------------------------
           s^4 + sqrt(2) BW s^3 + (2 w0^2 + BW^2) s^2 + sqrt(2) BW w0^2 s + w0^4
    """
    w1 = prewarping(f1, fs)
    w2 = prewarping(f2, fs)
    BW = w2 - w1
    w0_2 = w1 * w2
    num = [0.0, 0.0, BW ** 2, 0.0, 0.0]
    den = [1.0, np.sqrt(2) * BW, 2 * w0_2 + BW ** 2, np.sqrt(2) * BW * w0_2, w0_2 ** 2]
    return np.array(num), np.array(den)


def multiplicar_polinomios(p, q):
    """Producto de polinomios en z^-1 (convolucion de coeficientes)."""
    r = np.zeros(len(p) + len(q) - 1)
    for i in range(len(p)):
        for j in range(len(q)):
            r[i + j] += p[i] * q[j]
    return r


def potencia_polinomio(p, n):
    r = np.array([1.0])
    for _ in range(n):
        r = multiplicar_polinomios(r, p)
    return r


def bilineal(num, den, fs):
    """
    s = (2/T) (1 - z^-1) / (1 + z^-1),   2/T = 2 fs
    Se multiplica arriba y abajo por (1 + z^-1)^4; cada termino a_m s^m queda
        a_m (2 fs)^m (1 - z^-1)^m (1 + z^-1)^(4 - m)
    y se divide todo por a0 para que el denominador empiece en 1.
    """
    n = len(den) - 1
    c = 2 * fs
    b = np.zeros(n + 1)
    a = np.zeros(n + 1)
    for i in range(n + 1):
        m = n - i                                       # potencia de s
        termino = multiplicar_polinomios(potencia_polinomio([1.0, -1.0], m),
                                         potencia_polinomio([1.0, 1.0], n - m))
        b += num[i] * c ** m * termino
        a += den[i] * c ** m * termino
    return b / a[0], a / a[0]


def coeficientes_filtro():
    num, den = butterworth_pasabandas(cfg.CORTE_INFERIOR, cfg.CORTE_SUPERIOR, cfg.SAMPLE_RATE)
    return bilineal(num, den, cfg.SAMPLE_RATE)


B_FILTRO, A_FILTRO = coeficientes_filtro()


def ecuacion_diferencias(x, b, a):
    """y(n) = sum_{k=0}^{4} b_k x(n-k) - sum_{k=1}^{4} a_k y(n-k)"""
    b = [float(v) for v in b]
    a = [float(v) for v in a]
    orden = len(a) - 1
    xs = [0.0] * orden + [float(v) for v in x]
    ys = [0.0] * len(xs)
    for n in range(orden, len(xs)):
        acumulado = 0.0
        for k in range(orden + 1):
            acumulado += b[k] * xs[n - k]
        for k in range(1, orden + 1):
            acumulado -= a[k] * ys[n - k]
        ys[n] = acumulado
    return np.array(ys[orden:])


def preprocesar(audio):
    x = np.asarray(audio, dtype=float).reshape(-1)
    if not len(x) or not np.all(np.isfinite(x)):
        raise ValueError('El audio esta vacio o contiene muestras invalidas.')
    x = x - np.mean(x)                                  # quita la componente DC
    return ecuacion_diferencias(x, B_FILTRO, A_FILTRO)


def seleccionar_segmento(x):
    """Ventana de 3 s con mayor energia sum x(n)^2, buscada cada 0.25 s."""
    cantidad = int(cfg.SAMPLE_RATE * cfg.DURACION_SEGMENTO)
    if len(x) <= cantidad:
        return x
    acumulada = np.concatenate(([0.0], np.cumsum(x ** 2)))
    paso = int(cfg.SAMPLE_RATE * 0.25)
    inicios = list(range(0, len(x) - cantidad + 1, paso))
    if inicios[-1] != len(x) - cantidad:
        inicios.append(len(x) - cantidad)
    mejor, energia_mayor = 0, -1.0
    for i in inicios:
        energia = acumulada[i + cantidad] - acumulada[i]
        if energia > energia_mayor:
            mejor, energia_mayor = i, energia
    return x[mejor:mejor + cantidad]


# ============================================================================
# FFT POR SUBMUESTREO PAR / IMPAR
# ============================================================================
def fft_par_impar(x):
    """
    xp(n) = x(2n)    (pares)        xi(n) = x(2n+1)    (impares)
    G(k) = FFT de xp,  H(k) = FFT de xi   (N/2 puntos, misma regla)

        X(k)       = G(k) + W^k H(k)
        X(k + N/2) = G(k) - W^k H(k),     W = e^{-j 2 pi / N},  k = 0 .. N/2-1

    Se repite hasta llegar a 1 muestra: X(0) = x(0).
    x puede ser una matriz: cada fila es un bloque de N muestras.
    """
    x = np.asarray(x, dtype=complex)
    N = x.shape[-1]
    if N == 1:
        return x
    xp = x[..., 0::2]
    xi = x[..., 1::2]
    G = fft_par_impar(xp)
    H = fft_par_impar(xi)
    k = np.arange(N // 2)
    W = np.exp(-1j * 2 * np.pi * k / N)                 # W_N^k
    return np.concatenate([G + W * H, G - W * H], axis=-1)


def verificar_fft():
    """Compara la FFT par/impar con la DFT por definicion X(k) = sum x(n) e^{-j2pi kn/N}."""
    x = np.array([2, -1, 1, -1, 1, -1, 1, -1], dtype=float)
    n = np.arange(8)
    dft = np.array([np.sum(x * np.exp(-1j * 2 * np.pi * k * n / 8)) for k in range(8)])
    return float(np.max(np.abs(fft_par_impar(x) - dft)))

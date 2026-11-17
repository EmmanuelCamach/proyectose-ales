import numpy as np
from scipy.io import wavfile
from scipy.signal import bilinear, lfilter
from scipy.fft import dct, idct
import configuracion as cfg


def cargar_audio(ruta):
    sr, audio = wavfile.read(ruta)
    if sr != cfg.SAMPLE_RATE:
        raise ValueError(f'{ruta}: se requieren WAV a {cfg.SAMPLE_RATE} Hz.')
    # Escala del formato PCM; no se normaliza cada grabación por su amplitud.
    if np.issubdtype(audio.dtype, np.integer):
        info = np.iinfo(audio.dtype)
        centro = (info.max + 1) / 2 if info.min == 0 else 0
        escala = centro if info.min == 0 else -info.min
        audio = (audio.astype(float) - centro) / escala
    else:
        audio = audio.astype(float)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return audio


def coeficientes_fir():
    # hBP(m) = [sen(w2*m) - sen(w1*m)] / (pi*m).
    # hBP(0) = (w2 - w1) / pi; hd(n) = hBP(n) * ventana(n).
    if cfg.ORDEN_FIR < 2 or cfg.ORDEN_FIR % 2:
        raise ValueError('Use un orden FIR par y positivo.')
    n = np.arange(cfg.ORDEN_FIR + 1)
    m = n - cfg.ORDEN_FIR / 2
    w1 = 2 * np.pi * cfg.CORTE_INFERIOR / cfg.SAMPLE_RATE
    w2 = 2 * np.pi * cfg.CORTE_SUPERIOR / cfg.SAMPLE_RATE
    h = np.empty(len(n))
    h[m == 0] = (w2 - w1) / np.pi
    h[m != 0] = (np.sin(w2 * m[m != 0]) - np.sin(w1 * m[m != 0])) / (np.pi * m[m != 0])
    if cfg.VENTANA_FIR == 'hamming':
        ventana = 0.54 - 0.46 * np.cos(2 * np.pi * n / cfg.ORDEN_FIR)
    elif cfg.VENTANA_FIR == 'hanning':
        ventana = 0.5 - 0.5 * np.cos(2 * np.pi * n / cfg.ORDEN_FIR)
    else:
        raise ValueError('Ventana FIR: hamming o hanning.')
    return h * ventana, np.array([1.0])


def coeficientes_butterworth():
    # Prototipo H(p)=1/(p²+sqrt(2)*p+1).
    # BP: p=(s²+w0²)/(BW*s). Preajuste w=2*fs*tan(pi*f/fs).
    w1 = 2 * cfg.SAMPLE_RATE * np.tan(np.pi * cfg.CORTE_INFERIOR / cfg.SAMPLE_RATE)
    w2 = 2 * cfg.SAMPLE_RATE * np.tan(np.pi * cfg.CORTE_SUPERIOR / cfg.SAMPLE_RATE)
    bw = w2 - w1
    w0 = np.sqrt(w1 * w2)
    numerador = [bw ** 2, 0, 0]
    denominador = [1, np.sqrt(2) * bw, 2 * w0 ** 2 + bw ** 2,
                   np.sqrt(2) * bw * w0 ** 2, w0 ** 4]
    # s=2*fs*(1-z^-1)/(1+z^-1). Orden 2 del prototipo -> orden 4 BP.
    return bilinear(numerador, denominador, fs=cfg.SAMPLE_RATE)


def preprocesar(audio):
    x = np.asarray(audio, dtype=float).reshape(-1)
    if not len(x) or not np.all(np.isfinite(x)):
        raise ValueError('El audio está vacío o contiene muestras inválidas.')
    if not 0 < cfg.CORTE_INFERIOR < cfg.CORTE_SUPERIOR < cfg.SAMPLE_RATE / 2:
        raise ValueError('Se requiere 0 < corte inferior < corte superior < Nyquist.')
    if cfg.FILTRO == 'fir':
        b, a = coeficientes_fir()
        x = lfilter(b, a, x)  # Convolución FIR: y(n)=sum h(k)*x(n-k).
    elif cfg.FILTRO == 'butterworth':
        b, a = coeficientes_butterworth()
        x = lfilter(b, a, x)  # Ecuación en diferencias de H(z)=B(z)/A(z).
    elif cfg.FILTRO != 'ninguno':
        raise ValueError('Filtro: fir, butterworth o ninguno.')
    if not 0 <= cfg.COMPRESION_DCT < 100:
        raise ValueError('La compresión DCT debe estar entre 0 y menos de 100%.')
    if cfg.COMPRESION_DCT:
        # DCT-II ortonormal: anular los coeficientes de menor magnitud e invertir.
        coeficientes = dct(x, type=2, norm='ortho')
        cantidad = int(len(x) * cfg.COMPRESION_DCT / 100)
        coeficientes[np.argsort(np.abs(coeficientes))[:cantidad]] = 0
        x = idct(coeficientes, type=2, norm='ortho')
    return x


def seleccionar_segmento(x):
    """Selecciona tres segundos por energía; no utiliza la especie del audio."""
    cantidad = int(cfg.SAMPLE_RATE * cfg.DURACION_SEGMENTO)
    if len(x) <= cantidad:
        return x
    # Energía acumulada para comparar ventanas cada 0.25 segundos.
    acumulada = np.concatenate(([0.0], np.cumsum(x ** 2)))
    paso = max(1, int(cfg.SAMPLE_RATE * 0.25))
    inicios = np.arange(0, len(x) - cantidad + 1, paso)
    inicios = np.unique(np.append(inicios, len(x) - cantidad))
    energias = acumulada[inicios + cantidad] - acumulada[inicios]
    inicio = int(inicios[np.argmax(energias)])
    return x[inicio:inicio + cantidad]


def analizar(audio):
    x = seleccionar_segmento(preprocesar(audio))
    if len(x) < cfg.N_FFT:
        x = np.pad(x, (0, cfg.N_FFT - len(x)))
    bloques = x[:len(x) // cfg.N_FFT * cfg.N_FFT].reshape(-1, cfg.N_FFT)
    # DFT X(k)=sum x(n)*exp(-j*2*pi*k*n/N), calculada mediante FFT.
    espectros = np.fft.rfft(bloques, axis=1)
    limites = cfg.FRONTERAS_SUBBANDAS
    if not 3 <= len(limites) - 1 <= 8 or any(b <= a for a, b in zip(limites[:-1], limites[1:])):
        raise ValueError('Defina entre 3 y 8 subbandas no vacías.')
    if limites[0] != 0 or limites[-1] != espectros.shape[1]:
        raise ValueError('Las subbandas deben cubrir desde DC hasta Nyquist.')
    # Eij=(1/Mj)*sum |Xi(k)|². Primero energía, después promedio entre bloques.
    energias = np.array([np.mean(np.abs(espectros[:, a:b]) ** 2, axis=1)
                        for a, b in zip(limites[:-1], limites[1:])]).T
    vector = energias.mean(axis=0)
    magnitud = np.abs(espectros).mean(axis=0)
    frecuencias = np.fft.rfftfreq(cfg.N_FFT, 1 / cfg.SAMPLE_RATE)
    return vector, magnitud, frecuencias


def calcular_vector_energia(audio):
    return analizar(audio)[0]


def distancia_euclidiana(vector, referencia):
    return float(np.sqrt(np.sum((np.asarray(vector) - np.asarray(referencia)) ** 2)))


def reconocer(vector, referencias):
    distancias = {clave: distancia_euclidiana(vector, referencia)
                  for clave, referencia in referencias.items()}
    return min(distancias, key=distancias.get), distancias

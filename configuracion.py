from pathlib import Path

# Grabaciones: cantos de eBird / Macaulay Library reproducidos y grabados con el celular
#   Hormiguero Gorjinegro: https://ebird.org/species/bltant2
#   Turpial Amazonico:     https://ebird.org/species/orbtro3
SAMPLE_RATE = 48000          # fs (Hz)
DURACION_SEGMENTO = 3.0      # segundos de canto que se analizan (los de mayor energia)
DURACION_CAPTURA = 6.0       # segundos que se graban con el microfono
N_FFT = 2048                 # N = 2^11 (potencia de 2 para la FFT par/impar)
DELTA_F = SAMPLE_RATE / N_FFT    # Delta_f = fs / N = 23.4375 Hz

# Filtro Butterworth pasabandas (prototipo de orden 2 -> pasabandas de orden 4)
CORTE_INFERIOR = 200         # f1 (Hz)
CORTE_SUPERIOR = 10000       # f2 (Hz)

# Banco de filtros: 3 subbandas sobre X(0) .. X(1024)
#   X1 = X(0)   .. X(127)  ->     0 -  3000 Hz  (zona del Turpial)
#   X2 = X(128) .. X(255)  ->  3000 -  6000 Hz  (zona del Hormiguero)
#   X3 = X(256) .. X(1024) ->  6000 - 24000 Hz  (resto, hasta Nyquist)
FRONTERAS_SUBBANDAS = [0, 128, 256, N_FFT // 2 + 1]
LIMITES_HZ = [k * DELTA_F for k in FRONTERAS_SUBBANDAS[:-1]] + [SAMPLE_RATE / 2]

# Sesiones de grabacion: clips separados menos de UMBRAL_SESION segundos vienen
# de la misma grabacion original y nunca se reparten entre entrenamiento y prueba.
UMBRAL_SESION = 20
PLIEGUES = 5

# Microfono: energia media (1/N) sum x(n)^2 minima para considerar que hay canto
UMBRAL_SILENCIO = 1e-5

RUTA_BASE = Path(__file__).resolve().parent
RUTA_DATASET = RUTA_BASE / 'dataset'
RUTA_MODELO = RUTA_BASE / 'modelo_energias.json'
CARPETAS_ESPECIES = {'HG': 'Hormiguero_Gorjinegro', 'TA': 'Turpial_Amazonico'}
NOMBRES_ESPECIES = {'HG': 'Hormiguero Gorjinegro', 'TA': 'Turpial Amazónico'}

from pathlib import Path

SAMPLE_RATE = 48000  # Frecuencia de los WAV proporcionados.
DURACION_SEGMENTO = 3.0
N_FFT = 2048
FILTRO = 'butterworth'  # 'butterworth', 'fir' o 'ninguno'.
CORTE_INFERIOR = 200
CORTE_SUPERIOR = 10000
ORDEN_FIR = 100
VENTANA_FIR = 'hamming'  # 'hamming' o 'hanning'.
COMPRESION_DCT = 0  # Porcentaje de coeficientes anulados; 0 desactiva la compresión.
# Tres subbandas fijadas en la configuración. El repositorio no incluye el script ni los resultados de la selección de las fronteras.
# FFT positiva: [0:105], [105:255], [255:1025]. Incluye Nyquist.
FRONTERAS_SUBBANDAS = [0, 105, 255, N_FFT // 2 + 1]
LIMITES_HZ = [0, 105 * SAMPLE_RATE / N_FFT, 255 * SAMPLE_RATE / N_FFT, SAMPLE_RATE / 2]
RUTA_BASE = Path(__file__).resolve().parent
RUTA_DATASET = RUTA_BASE / 'dataset'
RUTA_MODELO = RUTA_BASE / 'modelo_energias.json'
CARPETAS_ESPECIES = {'HG': 'Hormiguero_Gorjinegro', 'TA': 'Turpial_Amazonico'}
NOMBRES_ESPECIES = {'HG': 'Hormiguero Gorjinegro', 'TA': 'Turpial Amazónico'}


def parametros():
    return {'SELECCION_SEGMENTO': 'maxima_energia_3s_v1', **{nombre: globals()[nombre] for nombre in (
        'SAMPLE_RATE', 'DURACION_SEGMENTO', 'N_FFT', 'FILTRO',
        'CORTE_INFERIOR', 'CORTE_SUPERIOR', 'ORDEN_FIR', 'VENTANA_FIR',
        'COMPRESION_DCT', 'FRONTERAS_SUBBANDAS')}}

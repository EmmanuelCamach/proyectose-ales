import configuracion as cfg
import json
import numpy as np
from configuracion import (SAMPLE_RATE, N_FFT, DURACION_SEGMENTO,
    FRONTERAS_SUBBANDAS, RUTA_MODELO, NOMBRES_ESPECIES)
from procesamiento import cargar_audio, analizar, reconocer


def cargar_modelo():
    if not RUTA_MODELO.exists():
        raise ValueError('Primero ejecute: python entrenamiento.py')
    modelo = json.loads(RUTA_MODELO.read_text(encoding='utf-8'))
    if modelo.get('parametros') != cfg.parametros():
        raise ValueError('Cambió la configuración. Ejecute otra vez entrenamiento.py.')
    print(f"Error en archivos reservados: {modelo['error_porcentaje']:.2f}%")
    if modelo['error_porcentaje'] > 5:
        print('Aviso: el modelo NO cumple el error máximo del 5%.')
    print('El sistema compara únicamente las dos especies entrenadas.')
    return modelo['referencias']


def procesar_senal(audio, referencias):
    vector, magnitud, frecuencias = analizar(audio)
    clave, distancias = reconocer(vector, referencias)
    print('Vector de energías:', np.round(vector, 4))
    for especie, distancia in distancias.items():
        print(f'{NOMBRES_ESPECIES[especie]}: distancia = {distancia:.4f}')
    print('Especie reconocida:', NOMBRES_ESPECIES[clave])
    return clave, vector, magnitud, frecuencias, distancias


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Reconocimiento mediante energía y distancia euclidiana.')
    parser.add_argument('--archivo', help='WAV para una prueba individual; sin esta opción usa micrófono.')
    args = parser.parse_args()
    referencias = cargar_modelo()
    if args.archivo:
        procesar_senal(cargar_audio(args.archivo), referencias)
        return
    import sounddevice as sd
    import matplotlib.pyplot as plt
    plt.ion()
    fig, ejes = plt.subplots(2, 1, figsize=(9, 7))
    print('Micrófono activo. Se analizan los 3 segundos de mayor energía de cada captura de 6 segundos. Ctrl+C para terminar.')
    cantidad = int(SAMPLE_RATE * (DURACION_SEGMENTO + 3))
    # El flujo permanece abierto: no se interrumpe la captura entre segmentos.
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32',
                            blocksize=cantidad) as microfono:
            while True:
                audio, desbordamiento = microfono.read(cantidad)
                if desbordamiento:
                    print('Se perdieron muestras. Repita el canto.')
                    continue
                if not np.any(audio):
                    print('Sin señal de audio.')
                    continue
                clave, vector, magnitud, frecuencias, distancias = procesar_senal(audio[:, 0], referencias)
                ejes[0].clear()
                ejes[0].plot(frecuencias, magnitud)
                for f in cfg.LIMITES_HZ:
                    ejes[0].axvline(f, color='gray', linestyle='--')
                ejes[0].set(xlabel='Frecuencia (Hz)', ylabel='Magnitud promedio',
                            title=f'Espectro filtrado: {NOMBRES_ESPECIES[clave]}')
                ejes[1].clear()
                ejes[1].bar([NOMBRES_ESPECIES[c] for c in distancias], list(distancias.values()))
                ejes[1].set(ylabel='Distancia euclidiana', title='Comparación con las dos especies')
                fig.tight_layout()
                plt.pause(0.01)
                if not plt.fignum_exists(fig.number):
                    break
    except KeyboardInterrupt:
        print('\nCaptura terminada.')
    finally:
        plt.close(fig)


if __name__ == '__main__':
    main()

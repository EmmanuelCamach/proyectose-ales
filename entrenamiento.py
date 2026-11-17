"""Ejecutar desde la raíz del repositorio: python entrenamiento.py
Después: python reconocimiento.py
Dependencias: pip install numpy scipy sounddevice matplotlib
Los archivos WAV se leen sin modificarlos.
Para evaluar individuos nuevos, reservar grabaciones de individuos distintos.
La partición por archivo actual no permite garantizar esa independencia.
"""
import configuracion as cfg
import json
import random
import numpy as np
from configuracion import (SAMPLE_RATE, N_FFT, DURACION_SEGMENTO,
    FRONTERAS_SUBBANDAS, RUTA_DATASET, RUTA_MODELO, CARPETAS_ESPECIES)
from procesamiento import cargar_audio, calcular_vector_energia, reconocer


def main():
    referencias = {}
    desviaciones = {}
    pruebas = []
    entrenamiento = []
    rng = random.Random(42)
    for clave, carpeta in CARPETAS_ESPECIES.items():
        archivos = sorted((RUTA_DATASET / carpeta).glob('*.wav'))
        if len(archivos) < 5:
            raise ValueError(f'{carpeta}: se necesitan al menos cinco archivos WAV.')
        rng.shuffle(archivos)
        cantidad_prueba = max(1, len(archivos) // 5)
        pruebas.extend((clave, archivo) for archivo in archivos[:cantidad_prueba])
        archivos_entrenamiento = archivos[cantidad_prueba:]
        vectores = [calcular_vector_energia(cargar_audio(p)) for p in archivos_entrenamiento]
        # Promedio de los vectores de energía, como EA y EB del documento.
        referencias[clave] = np.mean(vectores, axis=0).tolist()
        # Dispersión descriptiva; no cambia la distancia euclidiana del PDF.
        desviaciones[clave] = np.std(vectores, axis=0).tolist()
        entrenamiento.extend(str(p.relative_to(RUTA_DATASET)) for p in archivos_entrenamiento)

    resultados = []
    errores = 0
    for real, archivo in pruebas:
        predicha, distancias = reconocer(calcular_vector_energia(cargar_audio(archivo)), referencias)
        errores += predicha != real
        resultados.append({'archivo': str(archivo.relative_to(RUTA_DATASET)),
                           'real': real, 'predicha': predicha, 'distancias': distancias})
    error = 100 * errores / len(pruebas)
    modelo = {'parametros': cfg.parametros(), 'desviaciones': desviaciones,
              'sample_rate': SAMPLE_RATE, 'n_fft': N_FFT,
              'duracion': DURACION_SEGMENTO, 'fronteras': FRONTERAS_SUBBANDAS,
              'referencias': referencias, 'entrenamiento': entrenamiento,
              'pruebas': resultados, 'error_porcentaje': error,
              'advertencia': 'Evaluación de desarrollo por archivo; este conjunto ya se ha consultado durante ajustes anteriores. No es una prueba final independiente por individuo ni acredita procedencia de Xeno-canto.'}
    RUTA_MODELO.write_text(json.dumps(modelo, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'Evaluación de desarrollo: {errores} errores en {len(pruebas)} archivos ({error:.2f}%).')
    print('Error medido <= 5% en este conjunto.' if error <= 5 else 'NO cumple el error máximo del 5%.')
    print('Pendiente: prueba final con grabaciones nuevas de individuos independientes.')
    print('Verifique grabaciones de distintos individuos y registre la fuente original de cada una (por ejemplo, su enlace de Xeno-canto o de eBird/Macaulay Library).')
    print('Los nombres de WhatsApp del dataset no permiten comprobar su procedencia.')
    print(f'Modelo guardado en: {RUTA_MODELO}')
    for clave in referencias:
        print(clave, 'Promedio:', np.round(referencias[clave], 4))
        print(clave, 'Desviación estándar:', np.round(desviaciones[clave], 4))


if __name__ == '__main__':
    main()

"""
Obtiene las constantes EA y EB que usa reconocimiento.py y mide el error.

Ejecutar desde ProyectoFinal:   python entrenamiento.py

1. Calcula el vector de energias relativas Ex de cada grabacion (mismo algoritmo
   de reconocimiento.py: filtro, 3 s, FFT par/impar, 3 subbandas).
2. Agrupa los clips por sesion de grabacion (clips seguidos de una misma
   grabacion original de eBird) para que una grabacion nunca quede a la vez en
   entrenamiento y en prueba.
3. Validacion cruzada por sesiones (5 pliegues): cada grabacion se prueba una vez
   con EA y EB calculados sin su sesion ->  Error = errores / probadas * 100 %.
4. EA y EB finales = promedio de los Ex de todas las grabaciones de cada especie.
"""
import json
import re
import numpy as np
import configuracion as cfg
from procesamiento import cargar_audio, verificar_fft
from reconocimiento import extraer_energias, algoritmo_reconocimiento, EA, EB


def instante(nombre):
    """Hora escrita en el nombre 'WhatsApp Ptt ... at 9.24.36 PM.wav' (en segundos)."""
    m = re.search(r'at (\d+)\.(\d+)\.(\d+)\s*([AP]M)', nombre)
    if not m:
        return None
    h = int(m[1]) % 12 + (12 if m[4] == 'PM' else 0)
    return h * 3600 + int(m[2]) * 60 + int(m[3])


def listar_dataset():
    datos = {}
    for clave, carpeta in cfg.CARPETAS_ESPECIES.items():
        archivos = list((cfg.RUTA_DATASET / carpeta).glob('*.wav'))
        if len(archivos) < 5:
            raise ValueError(f'{carpeta}: se necesitan al menos 5 archivos WAV.')
        archivos.sort(key=lambda p: (instante(p.name) is None, instante(p.name) or 0, p.name))
        sesion, anterior, lista = 0, None, []
        for p in archivos:
            t = instante(p.name)
            if lista and (t is None or anterior is None or t - anterior > cfg.UMBRAL_SESION):
                sesion += 1
            lista.append((p, sesion))
            anterior = t
        datos[clave] = lista
    return datos


def distancia(Ex, E):
    return float(np.sqrt(np.sum((Ex - E) ** 2)))


def main():
    print(f'Verificacion FFT par/impar vs DFT: diferencia maxima = {verificar_fft():.2e}')
    datos = listar_dataset()
    Ex = {}
    for clave, lista in datos.items():
        print(f'Calculando energias de {cfg.NOMBRES_ESPECIES[clave]} ({len(lista)} grabaciones) ...')
        Ex[clave] = np.array([extraer_energias(cargar_audio(p))['Ex'] for p, _ in lista])

    # Validacion cruzada por sesiones
    resultados = []
    for f in range(cfg.PLIEGUES):
        prueba, ref = {}, {}
        for clave, lista in datos.items():
            sesiones = sorted({s for _, s in lista})
            grupo = set(sesiones[f::cfg.PLIEGUES])
            prueba[clave] = [i for i, (_, s) in enumerate(lista) if s in grupo]
            entren = [i for i, (_, s) in enumerate(lista) if s not in grupo]
            ref[clave] = Ex[clave][entren].mean(axis=0)
        for clave in datos:
            for i in prueba[clave]:
                dA = distancia(Ex[clave][i], ref['HG'])
                dB = distancia(Ex[clave][i], ref['TA'])
                predicha = 'HG' if dA < dB else 'TA'
                resultados.append({'archivo': datos[clave][i][0].name, 'real': clave,
                                   'predicha': predicha, 'pliegue': f + 1, 'dA': dA, 'dB': dB})
    errores = sum(r['real'] != r['predicha'] for r in resultados)
    error = 100.0 * errores / len(resultados)

    # Constantes finales con todas las grabaciones
    EA_nuevo = Ex['HG'].mean(axis=0)
    EB_nuevo = Ex['TA'].mean(axis=0)

    salida = {'fronteras_subbandas': cfg.FRONTERAS_SUBBANDAS, 'limites_hz': cfg.LIMITES_HZ,
              'grabaciones': {c: len(v) for c, v in datos.items()},
              'sesiones': {c: len({s for _, s in v}) for c, v in datos.items()},
              'EA': EA_nuevo.tolist(), 'EB': EB_nuevo.tolist(),
              'error_porcentaje': error, 'errores': errores, 'probadas': len(resultados),
              'pruebas': resultados}
    cfg.RUTA_MODELO.write_text(json.dumps(salida, indent=2, ensure_ascii=False), encoding='utf-8')

    print()
    for clave in datos:
        print(f"{cfg.NOMBRES_ESPECIES[clave]}: {salida['grabaciones'][clave]} grabaciones, "
              f"{salida['sesiones'][clave]} sesiones")
    print(f'\nValidacion cruzada por sesiones ({cfg.PLIEGUES} pliegues): '
          f'{errores} errores en {len(resultados)} grabaciones = {error:.2f} %')
    print('CUMPLE el error maximo del 5 %' if error <= 5 else 'NO CUMPLE el error maximo del 5 %')
    for r in resultados:
        if r['real'] != r['predicha']:
            print(f"  {r['real']} -> {r['predicha']}: {r['archivo']}")

    print('\nConstantes para reconocimiento.py:')
    print('EA = np.array([' + ', '.join(f'{v:.8f}' for v in EA_nuevo) + '])')
    print('EB = np.array([' + ', '.join(f'{v:.8f}' for v in EB_nuevo) + '])')
    if np.allclose(EA, EA_nuevo, atol=1e-6) and np.allclose(EB, EB_nuevo, atol=1e-6):
        print('(Coinciden con las constantes que ya tiene reconocimiento.py)')
        aciertos = sum(algoritmo_reconocimiento(cargar_audio(p))['clave'] == c
                       for c, lista in datos.items() for p, _ in lista)
        print(f'Con las constantes finales reconoce bien {aciertos} de {len(resultados)} grabaciones.')
    else:
        print('ATENCION: copie estas constantes en reconocimiento.py (EA y EB).')
    print(f'Resultados guardados en: {cfg.RUTA_MODELO}')


if __name__ == '__main__':
    main()

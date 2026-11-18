# Reconocimiento de cantos aviares por banco de filtros (energía en 3 subbandas)

Proyecto de procesamiento de señales (Universidad de los Llanos). Clasifica grabaciones de dos especies, **Hormiguero Gorjinegro (HG, especie A)** y **Turpial Amazónico (TA, especie B)**, con la energía relativa de tres subbandas de la FFT y la distancia euclidiana a un vector de referencia por especie.

## Cadena de procesamiento

1. Lectura del WAV con el módulo `wave` (PCM de 16 bits a 48 kHz; no se remuestrea). Si es estéreo se promedian los canales y se resta la componente DC.
2. Filtro Butterworth pasabanda de 200 Hz a 10 kHz (prototipo de 2.º orden, pasabanda de 4.º orden), con preajuste de frecuencia (*prewarping*) y transformación bilineal. La ecuación en diferencias está programada a mano; el código no usa SciPy.
3. Selección de los 3 s de mayor energía (ventanas cada 0,25 s).
4. FFT por submuestreo par/impar (radix-2) sobre bloques de N = 2048 muestras (Δf = 23,4375 Hz).
5. Tres subbandas: X(0–127) = 0–3000 Hz, X(128–255) = 3000–6000 Hz y X(256–1024) = 6000–24 000 Hz. Energía media por subbanda, promediada entre bloques y normalizada para que sume 1 (energías relativas, independientes de la ganancia).
6. Distancia euclidiana a las constantes `EA` y `EB` (definidas en `reconocimiento.py`) y decisión por mínima distancia. El algoritmo no tiene umbral de rechazo: siempre devuelve HG o TA. Solo en modo micrófono se descarta el silencio (`UMBRAL_SILENCIO`).

## Estructura

| Archivo | Contenido |
|---|---|
| `configuracion.py` | Parámetros: frecuencia de muestreo, FFT, filtro, subbandas, sesiones |
| `procesamiento.py` | Lectura, filtro Butterworth, selección de segmento y FFT par/impar |
| `reconocimiento.py` | Constantes `EA` y `EB`, extracción de energías y decisión; interfaz de línea de comandos |
| `entrenamiento.py` | Calcula `EA` y `EB` y mide el error por validación cruzada por sesiones |
| `interfaz.py` | Interfaz gráfica (tkinter y matplotlib) |
| `modelo_energias.json` | Registro de la última ejecución de `entrenamiento.py`; `reconocimiento.py` no lo lee |

## Uso

```bash
pip install -r requirements.txt
python reconocimiento.py --archivo ruta.wav   # clasifica un archivo
python reconocimiento.py --ejemplo-pdf        # ejemplo didáctico de 8 puntos
python reconocimiento.py --microfono          # captura en vivo (requiere sounddevice y PortAudio)
python reconocimiento.py --gui                # interfaz gráfica (también sin argumentos)
python entrenamiento.py                       # recalcula EA, EB y el error (requiere dataset/)
```

Ejecutar desde la raíz del repositorio. Si `entrenamiento.py` produce constantes distintas, hay que copiarlas en `reconocimiento.py`.

## Dataset

No está incluido en el repositorio. Para entrenar, cree `dataset/Hormiguero_Gorjinegro/` y `dataset/Turpial_Amazonico/` con archivos WAV PCM de 16 bits a 48 kHz.

## Resultados y límites

- 204 grabaciones (109 HG y 95 TA), agrupadas en 43 sesiones (11 HG y 32 TA) según la hora del nombre del archivo (separación máxima de 20 s dentro de una sesión).
- Validación cruzada por sesiones en 5 pliegues: **3 errores en 204 (1,47 %)**: dos HG clasificados como TA (de una misma sesión) y un TA clasificado como HG. Con las constantes finales se reconocen 201 de 204.
- Los cantos son de eBird / Macaulay Library reproducidos y grabados con el celular (notas de voz de WhatsApp), todos con fecha del 1 de junio de 2026 entre las 20:47 y las 21:47. Las sesiones se deducen de la hora del archivo, no de metadatos del individuo, y no hay un conjunto final independiente: el 1,47 % es una evaluación de desarrollo, no el desempeño esperado en campo.
- La separación entre especies proviene de las subbandas 1 y 2; la subbanda 3 sola da cerca de 47 % de error.
- Sin rechazo: ruido y tonos puros se asignan siempre a una especie. Un tono de 4 kHz queda a 0,042 de `EA`, igual que la distancia mediana de las grabaciones reales a su especie (0,041).

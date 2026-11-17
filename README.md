# Reconocimiento de cantos aviares por energía espectral en subbandas

Proyecto de procesamiento de señales (Ingeniería de Sistemas, Universidad de los Llanos). Clasifica grabaciones de dos especies, **Hormiguero Gorjinegro (HG)** y **Turpial Amazónico (TA)**, a partir de la energía media de tres subbandas de la FFT y la distancia euclidiana a un vector de referencia por clase.

## Cadena de procesamiento

1. Lectura del WAV (se exige 48 kHz; no se remuestrea) y conversión a mono.
2. Filtro Butterworth pasabanda de 200 Hz a 10 kHz (prototipo de 2.º orden, transformación bilineal con preajuste de frecuencia, `lfilter`).
3. Selección del segmento de 3 s con mayor energía (ventanas cada 0,25 s).
4. FFT por bloques de 2048 muestras (`numpy.fft.rfft`, sin ventana de ponderación).
5. Energía media por subbanda, con fronteras `[0, 105, 255, 1025]` (≈ 0, 2460,94, 5976,56 y 24 000 Hz), promediada entre bloques.
6. Clasificación por menor distancia euclidiana a las dos referencias. No hay umbral de rechazo: siempre devuelve HG o TA.

## Estructura

| Ruta | Contenido |
|---|---|
| `configuracion.py` | Parámetros, rutas y fronteras de subbandas |
| `procesamiento.py` | Carga, filtrado, selección de segmento y características |
| `entrenamiento.py` | Referencias por clase y evaluación sobre archivos reservados |
| `reconocimiento.py` | Clasificación de un WAV o captura desde micrófono |
| `modelo_energias.json` | Modelo entrenado (referencias, desviaciones, resultados) |
| `dataset/` | Grabaciones WAV por especie |
| `documentacion/` | Figuras del código y fórmulas del informe (SVG) |

## Uso

```bash
pip install -r requirements.txt
python entrenamiento.py                      # recalcula modelo_energias.json
python reconocimiento.py --archivo ruta.wav  # clasifica un archivo
python reconocimiento.py                     # micrófono (requiere sounddevice y PortAudio)
```

Ejecutar desde la raíz del repositorio. Si se cambia `configuracion.py`, hay que volver a entrenar.

## Resultados y límites

- Partición por archivo con semilla 42: 164 de entrenamiento y 40 reservados (21 HG, 19 TA).
- Resultado: 38 aciertos y 2 errores (5 %; ambos errores son TA clasificados como HG).
- **Es una evaluación de desarrollo**, no una validación independiente: el conjunto reservado se consultó durante los ajustes y la partición es por archivo, no por individuo, por lo que puede haber fuga entre entrenamiento y prueba.
- Las referencias HG y TA casi no se diferencian en la tercera subbanda (0,438 frente a 0,409); la separación viene de las dos primeras.
- Las características no se normalizan, así que dependen de la ganancia y la distancia al micrófono.

## Procedencia de los datos

Los nombres de archivo (`WhatsApp Ptt …`) no permiten comprobar de dónde provienen las grabaciones ni a qué individuos pertenecen. Antes de usar el modelo fuera de este trabajo, documente la fuente de cada audio (por ejemplo, enlace de Xeno-canto o de eBird/Macaulay Library) y reserve grabaciones de individuos distintos para una prueba final.

# Fórmulas del informe

Estas expresiones se conservan como imágenes vectoriales SVG para evitar los objetos de ecuación de Google Docs que provocaron problemas de carga. Los SVG se pueden abrir en navegador, insertar en documentos y escalar sin perder nitidez.

| Archivo | Expresión |
|---|---|
| [01_respuesta_en_frecuencia.svg](01_respuesta_en_frecuencia.svg) | Relación entre los espectros de entrada y salida: H(e^{jω}) = Y(e^{jω}) / X(e^{jω}). |
| [02_convolucion.svg](02_convolucion.svg) | Convolución discreta: y[n] = Σ x[k] h[n−k]. |
| [03_butterworth.svg](03_butterworth.svg) | Magnitud al cuadrado del prototipo Butterworth pasabajas de orden N. |
| [04_transformacion_bilineal.svg](04_transformacion_bilineal.svg) | Transformación bilineal usada para pasar del dominio analógico al digital. |
| [05_preajuste_frecuencia.svg](05_preajuste_frecuencia.svg) | Preajuste de las frecuencias de corte f₁ y f₂. |
| [06_transformada_discreta_fourier.svg](06_transformada_discreta_fourier.svg) | Definición de la DFT de N muestras. |
| [07_energia_media_subbandas.svg](07_energia_media_subbandas.svg) | Energía media de los bins de cada subbanda por bloque y promedio entre bloques. |

Las variables siguen la notación del informe y la implementación: x[n] (entrada), h[n] (respuesta al impulso), y[n] (salida), f_s (frecuencia de muestreo), f₁/f₂ (cortes), X_i[k] (DFT del bloque i), B_r (bins de la subbanda r), M_r (cantidad de bins) e I (cantidad de bloques).

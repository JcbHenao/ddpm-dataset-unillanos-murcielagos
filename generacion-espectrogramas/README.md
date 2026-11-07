# Generación de espectrogramas y selección de muestras

Scripts independientes (no usan `config.py`). Son anteriores al pipeline de
`limpieza-dataset/`, donde la generación actual está en `fase4` (pulsos) y `fase5` (secuencias).
Las rutas están al inicio de cada script.

## `pulsos/`

| Script | Qué hace |
| --- | --- |
| `generar_espectrogramas_pulsos_datos_anotados.py` | Espectrogramas de pulsos anotados con ventana fija de 20 ms. |
| `generar_espectrogramas_pulsos_datos_anotados_filtros.py` | Variante con ventana de 15 ms y filtro de banda (desde 20 kHz). |
| `seleccion_muestra_pulsos_molossus.py` | Selecciona la muestra de 1000 pulsos de alta calidad de las tres variantes de *Molossus* usada para los modelos de pulsos (`Muestra_Molossus_Alta_Calidad_128`). |

## `secuencias/`

| Script | Qué hace |
| --- | --- |
| `generar_espectrogramas_secuencia_bounding_box.py` | Espectrogramas de secuencias con las cajas de las anotaciones (tiempo y frecuencia) dibujadas encima. |
| `generar_espectrogramas_secuencia_con_filtros.py` | Secuencias con ventana de 1.5 s y filtro de banda. |
| `seleccion_scuencias.py` | Selección del conjunto de secuencias por especie para entrenamiento y validación. |
| `mover_secuencias.py` | Copia/mueve las secuencias seleccionadas a la carpeta de entrenamiento. |
| `elegir_secuencia.py` | Elige automáticamente las mejores secuencias y pulsos para figuras y diapositivas (ranking en `resultados/candidatas-diapositivas/`). |

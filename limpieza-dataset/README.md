# Limpieza del dataset y generación de espectrogramas (pipeline actual)

Los scripts de esta carpeta forman un solo pipeline y **se ejecutan desde la misma carpeta**:
`config.py`, `lector_anotaciones.py` y `espectrogramas_secuencias_comun.py` son importados por
las fases. En el servidor estaban en `/media/adalab/bodega/CamiloBuitrago/Generacion_Espectrogramas`.

| Orden | Script | Qué hace | Salida principal |
| --- | --- | --- | --- |
| – | `config.py` | Rutas, mapeos, parámetros de ventana y nombres de salida. | – |
| – | `lector_anotaciones.py` | Lectura de las tablas de selección de Raven (`.txt`) por anotador. | – |
| – | `espectrogramas_secuencias_comun.py` | Render común de secuencias (20–100 kHz, −80..0 dB, colormap jet) y ventanas. | – |
| 0 | `fase0_inventario_y_redundancia.py` | Inventario de fuentes de anotación y detección de redundancias. | `Fase0_Inventario/` |
| 1 | `fase1_auditoria_categorias_crudas.py` | Auditoría de valores crudos de especie, calidad y tipo de pulso contra `mapeos/`. | `Fase1_Auditoria_Categorias_Crudas/` |
| 2 | `fase2_normalizar_unificar.py` | Normaliza categorías y unifica todas las anotaciones (pulsos y secuencias). | `Fase2_Normalizado/anotaciones_unificadas.csv`, por especie, descartados y duplicados |
| 3 | `fase3_auditoria_resultado.py` | Revisa el resultado: pulsos y secuencias que exceden la ventana. | `Fase3_Auditoria_Resultado/` |
| 4 | `fase4_generar_espectrogramas.py` | Espectrogramas de pulsos (ventana de 20 ms). | `Espectrogramas_Pulsos_Anotados_512px/` |
| 5 | `fase5_generar_espectrogramas_secuencias.py` | Espectrogramas de secuencias en ventanas de 1.5 s. | `Espectrogramas_Secuencias_Chunks_1.5s_…/` |
| – | `contador-anotaciones.py` | Cuenta los archivos de anotación por nombre. | consola |

`mapeos/` contiene las tablas de equivalencia de nombres de especie, calidad y tipo de pulso
usadas en las fases 1 y 2.

Columnas de `anotaciones_unificadas.csv`: `Begin File`, `Begin Time (s)`, `End Time (s)`,
`Low Freq (Hz)`, `High Freq (Hz)`, `Peak Freq (Hz)`, `Delta Time (s)`, `Specie`, `Calidad`
(Alta/Media/Baja), `Pulse Type`, `Tipo_Anotacion` (Pulso/Secuencia), `Anotador`, `Anotacion_ID`.

## Versión previa

`version-previa/` guarda los scripts anteriores al pipeline (limpieza por separado de pulsos y
secuencias, normalización y auditorías de las anotaciones crudas en
`verificacion_datos_anotados/`). Sus salidas están en `datos/anotaciones/version-previa/`.

Requisitos: `numpy`, `pandas`, `librosa`, `soundfile`, `matplotlib`, `pillow`.

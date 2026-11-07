# Datos

Los datos del proyecto no se incluyen en el repositorio por su tamaño. Esta es la estructura de
la carpeta `datos/` que usan los scripts, con las salidas de los pipelines y las imágenes generadas.

| Carpeta | Contenido | Tamaño aprox. |
| --- | --- | --- |
| `anotaciones/Anotaciones_Unificadas/` | Salidas de las fases 0–3 de `limpieza-dataset/`, incluido `Fase2_Normalizado/anotaciones_unificadas.csv` y un CSV por especie. | 9 MB |
| `anotaciones/version-previa/` | Salidas de la limpieza previa (`Limpieza_Anotaciones_Pulsos/`) y de las auditorías de `verificacion_datos_anotados/`. | 7 MB |
| `espectrogramas/Espectrogramas_Pulsos_Anotados_512px/` | 7 828 espectrogramas de pulsos por especie con `metadatos.csv` (audio, tiempos, tipo de pulso, calidad). | 380 MB |
| `espectrogramas/Muestra_Molossus_Alta_Calidad_128/` | 1 000 pulsos de alta calidad de las tres variantes de *Molossus* usados en los modelos de pulsos, con `_muestra_seleccionada.csv`. | 29 MB |
| `generadas/pulsos-grid-timesteps-lr/<config>/` | 200 espectrogramas generados por cada configuración del barrido. | 94 MB |
| `candidatas-diapositivas/` | Imágenes de las secuencias y pulsos candidatos para figuras. | 24 MB |

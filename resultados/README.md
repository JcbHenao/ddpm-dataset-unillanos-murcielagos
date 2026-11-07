# Resultados

Solo métricas y tablas; las imágenes generadas están en `datos/` (no se suben).

## `pulsos-grid-timesteps-lr/`

Barrido de timesteps × learning rate del DDPM de pulsos de *Molossus* (128 × 128, 200 imágenes
generadas y 200 reales por configuración, 500 pasos de inferencia).

| Archivo | Contenido |
| --- | --- |
| `summary.csv` | Por configuración: mejor época, pérdida de validación y de test, ruta del mejor modelo. |
| `texto.txt` | Registro de la evaluación en test de otra corrida del barrido (mejor época ≤ 50). |
| `evaluation/evaluation_summary.csv` | FID, nearest SSIM real-real y sintético-sintético y prueba KS por configuración. |
| `evaluation/<config>/metrics/` | Las mismas métricas por configuración y la distribución completa de nearest SSIM. |
| `evaluation_nssim/nssim_summary.csv`, `nssim_tabla.tex` | Nearest SSIM cruzado (generada → real de test y de train). `baseline.json`: referencia real (test frente a train). |
| `evaluation_nssim/<config>/` | Valores de nSSIM por imagen generada. |

Resumen: el menor FID es `ts1500_lr2e4` (44.57) y el mayor nSSIM de test es `ts1500_lr1e4`
(0.1108 ± 0.0186), frente a 0.2065 ± 0.1041 de la referencia real.

## `secuencias-50-epocas/`

`resumen_metricas.csv`: IS, FID y SSIM por especie (500 generadas por especie) del experimento
previo de secuencias de 1024 × 128 px con 500 timesteps y 50 épocas.

## `candidatas-diapositivas/`

`ranking.csv`: secuencias candidatas para figuras, ordenadas por puntaje
(`generacion-espectrogramas/secuencias/elegir_secuencia.py`).

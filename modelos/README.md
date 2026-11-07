# Modelos

Todos usan `diffusers` (UNet2D o DiT) y `accelerate`. Las rutas a los datos y las carpetas de
salida están al inicio de cada script.

## `pulsos/` – espectrogramas de pulsos individuales (128 × 128)

| Script | Qué hace |
| --- | --- |
| `train-ddpm.py` | Script base del tutorial de `diffusers` (DDPM de 128 px), punto de partida. |
| `ddpm_pulsos.py` | DDPM no condicionado de pulsos de *Molossus*, 500 timesteps, 50 épocas. |
| `ddpm_grid_experimentos.py` | Barrido de timesteps (500, 1000, 1500) × learning rate (2.5e-5, 5e-5, 1e-4, 2e-4) con partición 600/200/200 (semilla 42) de la muestra de 1000 pulsos; guarda el mejor modelo por pérdida de validación. |
| `evaluar_grid.py` | Evalúa cada configuración del barrido: FID, nearest SSIM real-real y sintético-sintético con prueba KS. |
| `nssim_cruzado.py` | Nearest SSIM cruzado (cada generada contra su real más parecida), con el train como control de memorización; genera `nssim_tabla.tex`. |
| `ditmodel.py` | Variante con Diffusion Transformer (DiT) condicionado por especie sobre la misma muestra, 500 timesteps. |
| `imagen-func-modelo.py` | Figura ilustrativa del proceso forward y reverse del DDPM con el modelo entrenado. |

Resultados en `resultados/pulsos-grid-timesteps-lr/`.

## `secuencias/` – espectrogramas de secuencias (1.5 s)

| Script | Qué hace |
| --- | --- |
| `ddpm_secuencias_v1.py` | Primera versión: secuencias de 2048 × 128 px, 1500 timesteps, 100 épocas. |
| `evaluar_ddpm_secuencias.py` | Evaluación de la versión 1 (FID, IS, SSIM) con 1500 pasos de inferencia. |
| `train-ddpm-secuencias_v2.py` | Un modelo por especie, 1024 × 128 px, 500 timesteps, 50 épocas (experimento previo). |
| `train_ddpm_secuencias_condicional.py` | Un solo modelo condicionado por especie, mismas condiciones (experimento previo). |
| `train-ddpm-secuencias_v2.1.py` | Como v2 con `--epocas` y las dos GPU a la vez (cada GPU entrena un grupo de especies), con reanudación desde checkpoint. |
| `train_ddpm_secuencias_condicional2.py` | Como el condicionado con `--epocas`, entrenamiento distribuido en 2 GPU (`accelerate launch`), lote 8 y reanudación. |

`LEEME_paralelo.md` explica los cambios de las versiones 2.1 y condicional2 y los comandos.
Métricas del experimento previo en `resultados/secuencias-50-epocas/`.

## Pseudo-secuencias

El entrenamiento multicondicional (especie + número de pulsos + patrón) está en
`pseudosecuencias/` (`lanzar.py`, `configs/`, `ddpm/`), ver `pseudosecuencias/README_entrenamiento.md`.

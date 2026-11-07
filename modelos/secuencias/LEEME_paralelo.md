# Experimento previo de secuencias con más épocas y las 2 GPU

Son los mismos dos scripts del experimento previo (secuencias de 1024 × 128 px, 500 pasos de ruido,
fp16, lote 4, lr 1e-4, misma UNet y mismo dataset). Solo cambia lo marcado con `# PARALELO` en el
código:

| Cambio | Por qué |
| --- | --- |
| `--epocas` (por defecto 200, antes 50) y carpetas con sufijo `-e<epocas>` | Más entrenamiento sin sobrescribir los resultados de 50 épocas. |
| Condicionado: las 2 GPU entrenan **el mismo modelo** (DistributedDataParallel con `accelerate launch`) | Cada GPU procesa 2 de las 4 imágenes del lote y los gradientes se promedian. El lote total sigue siendo 4 y hay los mismos pasos por época que antes: el experimento es comparable, solo más rápido. |
| El learning rate avanza un paso por paso de optimizador | Sin esto, accelerate lo avanzaría una vez por GPU y el warmup y el coseno terminarían antes. |
| Por especie: cada GPU entrena **especies distintas al mismo tiempo** (`--grupo 0/2` y `--grupo 1/2`) | Son 6 modelos independientes; se reparten en 2 grupos con un número parecido de imágenes. |
| Checkpoint cada 30 épocas y reanudación automática | Con 200 épocas una corrida tarda días; si se corta, se relanza el mismo comando y sigue. |
| Métricas del condicionado con `sync_on_compute=False` | **Error del script original con 2 GPU:** cada GPU evalúa una especie distinta y torchmetrics junta los datos de ambas al calcular, así que el FID e IS de una especie mezclaban la de la otra GPU. Con una GPU (como se corrió antes) no pasaba. |

## Comandos

Desde la carpeta donde se corrieron los scripts originales (las salidas se crean ahí, con
`-e200` en el nombre), con el entorno que ya tiene PyTorch para CUDA 12.8:

```bash
source /media/adalab/bodega/CamiloBuitrago/pseudosecuencias/.venv-pseudo/bin/activate

# 1. Condicionado, las 2 GPU en el mismo modelo
tmux new -s condicional
accelerate launch --multi_gpu --num_processes 2 --mixed_precision fp16 \
    train_ddpm_secuencias_condicional.py --epocas 200 2>&1 | tee condicional_e200.log
#   (Ctrl+B y luego D para salir de tmux sin cortarlo; tmux attach -t condicional para volver)

# 2. Cuando termine: un modelo por especie, un grupo de especies en cada GPU
tmux new -s gpu0
CUDA_VISIBLE_DEVICES=0 python train-ddpm-secuencias_v2.py --epocas 200 --grupo 0/2 2>&1 | tee especies_gpu0.log
tmux new -s gpu1
CUDA_VISIBLE_DEVICES=1 python train-ddpm-secuencias_v2.py --epocas 200 --grupo 1/2 2>&1 | tee especies_gpu1.log
```

El script imprime al inicio cómo quedaron los grupos, por ejemplo
`grupo 0/2: ['MolossusMolossus', ...] (N imagenes)`.

## Salidas

| Qué | Dónde |
| --- | --- |
| Condicionado | `ddpm-secuencias-1024x128-500ntt-condicional-e200/` (`samples/`, `unet/`, `scheduler/`, `generadas/`, `metricas/`, `resumen_metricas.csv`) |
| Por especie | `ddpm-secuencias-1024x128-500ntt-e200-<especie>/` (`samples/`, modelo, `generadas/`, `metricas.json`) |
| Resumen por especie | `resumen_metricas-e200.csv`: junta las 6 especies de los dos grupos |
| Reanudar | `checkpoint/` y `progreso.json` dentro de cada carpeta |

Si un proceso se corta, se relanza **el mismo comando**: sigue desde el último checkpoint, y en el
script por especie se saltan las especies que ya tienen `metricas.json`.

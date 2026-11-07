# Entrenamiento DDPM – espectrogramas de murciélagos

Código para entrenar los modelos de difusión con experimentos definidos en archivos YAML y una
cola que ejecuta varios entrenamientos en orden, cada uno en su propia carpeta. En el servidor
está en la misma carpeta que las pseudo-secuencias (`pseudosecuencias/`) y usa el mismo entorno
`.venv-pseudo`.

## Estructura

```
pseudosecuencias/            # (parte de entrenamiento)
├── configs/
│   ├── base.yaml                  # valores por defecto; todo experimento hereda de aquí
│   ├── base_pseudo.yaml           # base para las pseudo-secuencias (multicondicional, 128x512)
│   ├── experimentos/              # un YAML por experimento (solo lo que cambia)
│   └── experimentos_pseudo/       # experimentos con pseudo-secuencias
├── ddpm/                          # código de entrenamiento (datos, ruido, modelo, métricas…)
├── verificar_entorno.py           # revisa entorno, datos y memoria ANTES de entrenar
├── lanzar.py                      # ejecuta una cola de experimentos
├── resumir.py                     # junta métricas de todas las corridas en runs/resumen.csv
├── generar.py                     # genera imágenes con un modelo ya entrenado
├── requirements-entrenamiento.txt / environment.yml
└── runs/                          # se crea al entrenar: una carpeta por corrida
```

## 1. Instalación (una sola vez en el servidor)

```bash
cd pseudosecuencias

# Opción A: conda
conda env create -f environment.yml
conda activate ddpm-murcielagos

# Opción B: venv
python3 -m venv .venv && source .venv/bin/activate

# PyTorch según la versión de CUDA del servidor (ver https://pytorch.org/get-started/locally/)
nvidia-smi                       # la esquina superior derecha dice la versión de CUDA del driver
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126   # driver 12.6 o mayor

pip install -r requirements-entrenamiento.txt

# Comprobar que PyTorch ve la GPU (debe imprimir True)
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

**Importante:** `pip install torch` sin `--index-url` instala la versión para la CUDA más nueva
(p. ej. `+cu130`). Si el driver es más antiguo, PyTorch muestra
`The NVIDIA driver on your system is too old` y **entrena en CPU**. La CUDA de PyTorch
(`torch.version.cuda`) tiene que ser igual o menor que la que muestra `nvidia-smi`. Por eso
`entrenamiento.permitir_cpu` es `false`: sin GPU la corrida falla de inmediato en lugar de
tardar días, y `verificar_entorno.py` marca ese caso como error.

El entorno queda separado del que usaste para los scripts anteriores. Cada corrida guarda en
`entorno.json` las versiones exactas de los paquetes, la GPU y el host, para poder reproducirla.

## 2. Verificar antes de entrenar

```bash
python verificar_entorno.py configs/experimentos/
```

Para cada experimento revisa:

- que la configuración sea válida (por ejemplo, que la resolución sea divisible por la UNet);
- cuántas imágenes hay por especie;
- **el tamaño real de las imágenes** contra el configurado: si no coinciden, es un error, porque
  redimensionar cambia la escala de tiempo y frecuencia;
- cuánta señal queda en el último paso de ruido (debería ser ~0 %);
- un paso de entrenamiento real en la GPU, con la memoria pico y una estimación de las horas.

El reporte queda en `verificacion_entorno.txt`. Si hay errores, el script termina con código 1.

## 3. Orden recomendado de experimentos

| Experimento | Para qué |
| --- | --- |
| `00_pulsos_referencia` | **Control del entorno.** Repite la configuración ganadora del artículo (ts1500, lr 1e-4). Si el FID sale muy distinto de ~44–48, algo del entorno está mal, no del modelo. Ajusta antes la ruta de datos en el YAML. |
| `01_sobreajuste_secuencias` | 16 secuencias por especie y muchas épocas. Si ni así aparecen pulsos, el problema es de ruido o de representación. |
| `02_secuencias_cond_lineal1500` | Condicionado por especie, lineal con 1500 pasos, EMA. |
| `03_secuencias_cond_coseno_zsnr` | Condicionado, coseno con SNR terminal cero, predicción v y escala de entrada 0.35. |
| `04_secuencias_por_especie_lineal1500` | Un modelo por especie (se expande en 6 corridas). |

## 3b. Pseudo-secuencias con varias condiciones

Los experimentos de `configs/experimentos_pseudo/` entrenan con el dataset del proyecto
`pseudosecuencias` (P4) en modo `multicondicional`: un solo modelo condicionado a la vez por
**especie**, **número de pulsos** (`n`: 2 o 3) y **patrón de frecuencia** (`patron`: `alto-bajo`,
`bajo-alto`, `alto-bajo-alto`, `bajo-alto-bajo`, `constante`, `irregular`). Heredan de
`configs/base_pseudo.yaml`, que apunta a
`pseudosecuencias/datos/P4_dataset/metadatos.csv` y usa imágenes de
128 × 512 px.

| Experimento | Para qué |
| --- | --- |
| `20_pseudo_sobreajuste` | Control 2: 16 imágenes por especie; debe memorizar pulsos, espaciado y patrón. |
| `21_pseudo_multicond_lineal1500` | Modelo principal: especie + n + patrón, lineal 1500 pasos, EMA. |
| `22_pseudo_multicond_coseno_zsnr` | Como 21 con coseno, SNR terminal cero, predicción v y escala 0.5. |
| `23_pseudo_sin_patron` | Ablación de 21 sin la condición de patrón. |

Cómo funciona (`ddpm/modelo.py`):

- Cada condición tiene su propio embedding; se suman y entran a la UNet junto con el embedding del
  paso de ruido. Una combinación poco frecuente aprovecha lo aprendido de cada condición por separado.
- `entrenamiento.prob_sin_condicion` (0.1 por defecto) borra al azar cada condición durante el
  entrenamiento. Así se puede generar fijando solo algunas condiciones y usar guía sin
  clasificador (`evaluacion.guia` o `generar.py --guia`).
- La lista `condiciones` del YAML elige qué columnas de `metadatos.csv` se usan; quitar una es
  una ablación (como hace el experimento 23).
- Las métricas se calculan por especie; las imágenes generadas reciben las combinaciones de `n` y
  `patron` de las imágenes reales de esa especie.
- La grilla `muestras/epoca_XXXX.png` tiene una fila por combinación frecuente; el
  `epoca_XXXX.txt` del mismo nombre dice qué combinación es cada fila.
- Si `metadatos.csv` no tiene la columna `patron` (renderizado con una versión anterior de
  `p3_grupos.py`), se calcula a partir de `transiciones` con el mismo criterio y se avisa.
- Con `especies: []` entran solo las especies con al menos `datos.min_imagenes_por_especie`
  (50) imágenes en train; las demás se informan como descartadas. Para forzar una especie,
  escríbela en la lista `especies`.

```bash
python verificar_entorno.py configs/experimentos_pseudo/
python lanzar.py configs/experimentos_pseudo/20_pseudo_sobreajuste.yaml   # primero el Control 2
python lanzar.py configs/experimentos_pseudo/                             # el resto, si el 20 pasa
python generar.py runs/21_pseudo_multicond_lineal1500 --especie MolossusMolossus \
    --pulsos 3 --patron alto-bajo-alto --cantidad 32
```

## 4. Lanzar

```bash
# Un experimento
python lanzar.py configs/experimentos/00_pulsos_referencia.yaml

# Toda la carpeta, en orden alfabético
python lanzar.py configs/experimentos/

# Ver la cola sin ejecutar
python lanzar.py configs/experimentos/ --listar

# Dos GPUs en la misma corrida
python lanzar.py configs/experimentos/02_secuencias_cond_lineal1500.yaml --gpus 2

# Dos colas en paralelo, una por GPU (en dos terminales o sesiones de tmux)
python lanzar.py configs/experimentos/02_*.yaml --dispositivos 0
python lanzar.py configs/experimentos/03_*.yaml --dispositivos 1
```

Para que siga corriendo al cerrar la sesión SSH, lánzalo dentro de `tmux` o con
`nohup python lanzar.py ... > cola.log 2>&1 &`.

**Reglas de la cola** (el `nombre` del YAML identifica la corrida):

| Situación | Qué hace |
| --- | --- |
| No existe | La entrena. |
| Completada, misma configuración | La salta (`--repetir` para volver a correrla). |
| A medias, misma configuración | La reanuda desde el último checkpoint. |
| Ya existe con **otra** configuración | No arranca la cola: cambia el `nombre` en el YAML, o usa `--desde-cero`. |
| Falla | Registra el error en `estado.json` y sigue con la siguiente. |

`--desde-cero` y `--repetir` nunca borran: mueven la corrida anterior a `runs/_archivo/`.

## 5. Qué queda en cada corrida

```
runs/<nombre>/
├── config.yaml              # configuración completa ya resuelta (sin herencia)
├── entorno.json             # versiones, GPU, host, commit de git
├── estado.json              # en_cola / en_curso / completado / fallido / interrumpido, pasos, error
├── salida.log               # todo lo que imprimió el entrenamiento
├── perdidas_por_epoca.csv
├── muestras/epoca_XXXX.png  # grilla de muestras (una fila por especie si es condicional)
├── checkpoints/ultimo/      # para reanudar
├── modelo/unet, modelo/scheduler, modelo/clases.json   # pesos finales (EMA si se usó)
├── muestras_finales/              # secuencias de prueba para mostrar (ver abajo)
├── metricas/generadas/<especie>/   # imágenes usadas para las métricas
└── metricas.json / metricas.csv
```

**Secuencias de prueba para mostrar** (`muestras_finales/`): al terminar cada corrida se generan
`evaluacion.muestras_finales.por_combinacion` imágenes (4 por defecto) para cada combinación de
condiciones (especie, n, patrón) y se copian otras tantas reales de train con la misma combinación:

```
muestras_finales/
├── hoja_<especie>.png                      por combinación: fila de reales y fila de generadas
├── <especie>/n3_alto-bajo-alto/generada_00.png … real_00.png …
└── indice.csv                              tipo, condiciones, archivo e imagen real de origen
```

Para una corrida ya terminada (o para pedir más imágenes): `python presentar.py runs/<nombre>
--por-combinacion 8`.

`runs/resumen.csv` junta todas las corridas: estado, parámetros clave y métricas por especie.
Se actualiza al final de cada cola, o a mano con `python resumir.py`.

Curvas de pérdida y tasa de aprendizaje: `tensorboard --logdir runs`.

## 6. Agregar un experimento

Crea un YAML en `configs/experimentos/` que herede de la base y cambie solo lo necesario:

```yaml
hereda: ../base.yaml
nombre: 05_secuencias_cond_lineal2000_lr5e-5   # nombre único: es la carpeta en runs/
descripcion: Lo mismo que 02 con 2000 pasos y lr menor.
ruido:
  pasos_entrenamiento: 2000
entrenamiento:
  lr: 5.0e-5
```

Las claves disponibles y su explicación están en `configs/base.yaml`.

## 7. Generar con un modelo entrenado

```bash
python generar.py runs/02_secuencias_cond_lineal1500 --especie SBilineata --cantidad 100
python generar.py runs/04_secuencias_por_especie_lineal1500-SLeptura --cantidad 50 --ddim 100
# multicondicional: cada condicion es opcional; --guia > 1 refuerza las condiciones pedidas
python generar.py runs/21_pseudo_multicond_lineal1500 --especie SLeptura --patron bajo-alto --guia 2
```

## 8. Notas sobre los datos de entrada

Al revisar los scripts de generación de espectrogramas aparecieron cuatro puntos que afectan
el entrenamiento:

1. **`SOBREMUESTREO_FRAMES` no está definido** en `espectrogramas_secuencias_comun.py`, pero
   `calcular_hop_length` lo usa: la fase 5 fallaría con `NameError` salvo que se defina en otro lado.
2. **Ancho de las secuencias.** El módulo común usa `ANCHO_IMG_PX = 2048`, mientras que los
   scripts de entrenamiento leían `Espectrogramas_Secuencias_1024x128px` con
   `Resize((128, 1024))`. Si esas imágenes salieron a 2048 px y se redimensionaron, los pulsos
   quedaron a la mitad de ancho. `verificar_entorno.py` ahora detecta cualquier diferencia de tamaño.
3. **Eje de frecuencia inconsistente.** Cuando un audio tiene Nyquist menor que `HIGH_FREQ`, el
   recorte tiene menos bandas y luego se estira a 128 px: la misma fila de píxeles corresponde a
   kHz distintos según el audio. Para un modelo que debe aprender la alternancia de frecuencias,
   conviene rellenar las bandas faltantes con el piso de −80 dB en lugar de estirar.
4. **Pulsos y secuencias se renderizan distinto.** Los pulsos usan `specshow` de matplotlib
   (20–120 kHz, escala de color que puede variar por imagen) y las secuencias usan el render
   uniforme (20–100 kHz, −80..0 dB fijo). Para combinar ambos datasets en pseudo-secuencias hay
   que volver a renderizar los dos con la misma función.

Lo que sí está bien: los `metadatos.csv` de pulsos y secuencias guardan `audio_origen` y los
tiempos, y el archivo maestro tiene `Begin Time`, `End Time`, `Begin File` y `Anotacion_ID` de
cada pulso. Con eso se pueden vincular pulsos y secuencias por grabación sin anotar nada nuevo.

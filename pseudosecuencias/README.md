# Pseudo-secuencias de 2 y 3 pulsos

Construye pseudo-secuencias a partir de las anotaciones que ya existen: 2 o 3 pulsos
consecutivos del mismo audio, con el silencio real entre ellos y todas del mismo tamaño. **No vuelve a anotar ni a
limpiar nada**: lee las rutas y utilidades del pipeline de limpieza (`config.py` y
`espectrogramas_secuencias_comun.py`) y escribe sus salidas en una carpeta propia.

El cronograma y la razón de cada etapa están en la pestaña «Pseudo-secuencias» del documento
del proyecto.

## Tres entornos, cada uno con su papel

| Entorno | Para qué | Dónde |
| --- | --- | --- |
| El del pipeline de limpieza | Fases 0–6 (maestro, espectrogramas) | Ya existe; no se toca |
| `.venv-pseudo` (este proyecto) | Vincular, medir intervalos, renderizar y armar el dataset | `bash crear_entorno.sh` |
| El de `entrenamiento_ddpm` | Entrenar y evaluar con PyTorch | Ya existe |

Este entorno no instala PyTorch: solo lo necesario para procesar audio y datos, y para poder
importar `config.py` del pipeline.

## Dónde está cada cosa en el servidor

```
/media/adalab/bodega/CamiloBuitrago/                  RUTA_BASE (config.py, o ANOTACIONES_BASE)
├── anotaciones/Anotaciones/{Pulsos,Secuencias}/      fuente canónica (tablas de Raven por anotador)
├── audios/                                           .wav, en subcarpetas (RUTA_AUDIOS)
├── Generacion_Espectrogramas/                        pipeline de limpieza (pipeline_dir)
│   ├── config.py, espectrogramas_secuencias_comun.py, lector_anotaciones.py, fase0…fase5
│   ├── mapeos/                                       mapeo_especies, mapeo_calidad, mapeo_pulse_type
│   ├── Anotaciones_Unificadas/Fase2_Normalizado/
│   │   └── anotaciones_unificadas.csv                ARCHIVO_MAESTRO: lo único que lee este proyecto
│   └── Pseudosecuencias/                             salidas de versiones anteriores (ya no se usan)
└── pseudosecuencias/                                 ESTE proyecto (código + .venv-pseudo)
    ├── datos/                                        SALIDAS: P0…P4 (el dataset) y C1
    └── runs/                                         entrenamientos (modelos, muestras, métricas)
```

Este proyecto **solo lee** el archivo maestro y los audios, y toma las rutas de `config.py`. No
lee las tablas de Raven directamente: la fase 2 del pipeline ya las normalizó (especies, calidad,
duplicados exactos), así que se parte de su resultado.

## Comandos para configurar el entorno (una sola vez)

```bash
# 1. Copiar el proyecto al servidor y descomprimirlo en CamiloBuitrago/
cd /media/adalab/bodega/CamiloBuitrago
unzip pseudosecuencias.zip              # crea ./pseudosecuencias
cd pseudosecuencias

# 2. Comprobar Python (3.9 o más nuevo) y crear el entorno virtual
python3 --version
bash crear_entorno.sh                   # crea .venv-pseudo e instala requirements.txt
                                        # y guarda las versiones exactas en requirements.lock.txt

# 3. Activarlo (en cada sesión nueva de terminal)
source /media/adalab/bodega/CamiloBuitrago/pseudosecuencias/.venv-pseudo/bin/activate

# 4. Verificar entorno, rutas y datos antes de procesar nada
python p0_verificar.py
```

Si `crear_entorno.sh` dice que falta `venv` (Ubuntu sin `python3-venv`) y no hay permisos de
administrador, el entorno se puede crear con conda:

```bash
conda create -y -p /media/adalab/bodega/CamiloBuitrago/pseudosecuencias/.venv-pseudo python=3.11
conda activate /media/adalab/bodega/CamiloBuitrago/pseudosecuencias/.venv-pseudo
pip install -r requirements.txt && pip freeze > requirements.lock.txt
```

Para usar otra versión de Python: `PYTHON=python3.11 bash crear_entorno.sh`. Para recrear el
entorno desde cero: `rm -rf .venv-pseudo && bash crear_entorno.sh`.

`pipeline_dir` en `parametros.yaml` ya apunta a `Generacion_Espectrogramas`. Si el pipeline se
mueve, se ajusta ahí o con `export PIPELINE_LIMPIEZA_DIR=/ruta/donde/esta/config.py`. La variable
`ANOTACIONES_BASE` del pipeline sigue funcionando igual para cambiar la ruta base.

## Etapas

```bash
python p0_verificar.py      # entorno, rutas, columnas del maestro, audios y frecuencias de muestreo
python p1_vincular.py       # orden de los pulsos por audio, secuencia a la que pertenecen, intervalos
python p2_intervalos.py     # frecuencia pico por pulso, intervalos por especie y transición
python p3_grupos.py         # pares y tríos, ventana única, partición por grabación
python p4_render.py         # imágenes del dataset (--sobrescribir para rehacerlo)
python c1_revisar.py        # Control 1: comprobaciones y hojas de contacto antes de entrenar
```

Cada etapa deja un reporte `.txt` junto a sus salidas en
`pseudosecuencias/datos/` (se cambia con `salida` en `parametros.yaml`):

```
datos/
├── P0_verificacion/verificacion.txt
├── P1_vinculacion/      pulsos_vinculados.csv, resumen_vinculacion.csv, vinculacion.txt
├── P2_intervalos/       pulsos_con_frecuencia.csv, intervalos_por_especie.csv,
│                        frecuencias_por_especie.csv, histogramas/, intervalos.txt
├── P3_grupos/           grupos.csv (incluidos, excluidos y motivo), ventana.json, grupos.txt
├── P4_dataset/
│   ├── train/<especie>/*.png
│   ├── val/<especie>/*.png
│   ├── test/<especie>/*.png
│   ├── metadatos.csv            una fila por imagen (lo que lee el entrenamiento: configs/base_pseudo.yaml)
│   └── render.txt
└── C1_revision/         revision.txt (lista para aprobar), hojas/<especie>.png
```

## Cómo se construye cada pseudo-secuencia

- **Grupo:** n pulsos consecutivos del mismo audio y especie (n = 2 o 3), con enlaces válidos
  según P2 (misma especie, sin solapamiento, dentro de la misma secuencia anotada) y cada pulso
  con una calidad aceptada (`grupos.calidades`, por defecto Alta y Media).
- **El espaciado no se filtra:** la distancia entre pulsos, con toda su variabilidad, es parte
  de lo que el modelo tiene que aprender junto con la forma de los pulsos. Por eso no se recortan
  intervalos por percentiles (`grupos.filtrar_por_percentil: false`); solo se excluyen los grupos
  que no caben en la ventana, y `c1_revisar.py` mide cuánto sesga eso la distribución.
- **Ventana única:** todas las imágenes, de 2 y de 3 pulsos, miden lo mismo (128 × 512 px por
  defecto) y cubren la misma duración. La duración sale del cuantil 0.95 de la duración de los
  tríos, más los márgenes; los grupos que no caben se excluyen y quedan registrados en `grupos.csv`.
- **Silencio real:** la imagen es audio real desde 5 ms antes del primer pulso hasta 5 ms
  después del último, así que el espacio entre pulsos se ve tal como es. El grupo va alineado
  a la izquierda y las columnas que sobran a la derecha quedan en el piso (−80 dB). No se usa
  audio real después del último pulso, porque ahí aparecería el pulso siguiente y un par
  parecería un trío.
- **Escala fija:** todos los audios se remuestrean a 320 kHz. Con `n_fft` 512 cada fila mide
  625 Hz, así que 20–100 kHz son exactamente 128 filas, y con un hop fijo cada columna mide
  los mismos milisegundos. No se redimensiona nada. En audios con Nyquist menor a 100 kHz, la
  banda alta queda en el piso en vez de estirarse.
- **Patrón de frecuencia:** cada grupo se etiqueta según cómo cambia la frecuencia pico entre
  pulsos (umbral de 2 kHz): `alto-bajo` y `bajo-alto` en pares, `alto-bajo-alto` y
  `bajo-alto-bajo` en tríos, `constante` si no cambia e `irregular` en cualquier otro caso. Junto
  con la especie y el número de pulsos, es una de las condiciones del modelo (columna `patron` de
  `grupos.csv` y `metadatos.csv`).
- **Partición por grabación:** todos los grupos de un audio van a la misma partición. Las
  grabaciones de cada especie se reparten para acercarse a 70/15/15 en número de grupos.

### Qué cuenta como intervalo válido

Dos pulsos consecutivos del mismo audio, de la misma especie, sin solaparse y dentro de la
misma secuencia anotada. Si ninguno de los dos está en una secuencia, cuentan solo si están a
menos de `max_intervalo_sin_secuencia_s` (0.5 s por defecto), para no medir pausas entre pasadas.

Los pulsos se ordenan **todos**, antes de cualquier filtro de calidad. Si se filtrara antes, el
intervalo entre dos pulsos de alta calidad con uno de baja calidad en medio sería el doble del real.

### Por qué se separa por transición

En especies que alternan frecuencia, la distribución de intervalos tiene dos modos: dentro
del par y entre pares. La mediana de todos los intervalos cae entre los dos y no describe a
ninguno. Por eso cada intervalo se etiqueta según si la frecuencia sube, baja o se mantiene
respecto al pulso anterior (umbral de 2 kHz, criterio de Jung et al. 2014).

## Controles sobre los datos del pipeline

- **Tiempos:** `Begin Time (s)` es relativo al `.wav` de `Begin File` en la fuente canónica (lo
  documenta `fase0`). `p2_intervalos.py` lo comprueba: compara la frecuencia pico que mide en cada
  recorte con el `Peak Freq (Hz)` que guardó Raven; si menos del 80 % coincide a 2 kHz, avisa.
- **Pulsos anotados dos veces:** la fase 2 solo elimina duplicados exactos. Si Jancel y Maira
  marcaron el mismo pulso con tiempos casi iguales, `p1_vincular.py` conserva uno (el de mejor
  calidad) y deja los otros en `pulsos_doble_anotacion.csv`. Sin esto, los dos aparecerían
  solapados y cortarían la cadena de pulsos consecutivos.
- **Nombres de audio repetidos:** `indexar_audios` del pipeline indexa por nombre de archivo, así
  que dos `.wav` con el mismo nombre en carpetas distintas se pisan. `p0_verificar.py` los lista.
- **Especies:** son las de `mapeos/mapeo_especies.csv` ya normalizadas por la fase 2
  (`MolossusMolossus`, `SBilineata`, …), que son también los nombres de carpeta del dataset.

## Antes de entrenar

`c1_revisar.py` comprueba automáticamente el tamaño único, que no haya grabaciones en dos
particiones, que cada especie tenga train/val/test y que los intervalos del dataset representen
los de todos los pares. Las hojas de contacto marcan el inicio anotado de cada pulso con un
triángulo blanco y el fin del audio real con una línea: sirven para revisar a ojo que el silencio
entre pulsos y la alternancia de frecuencia se vean bien. Si `p3_grupos.py` avisa que los pulsos de
alguna especie miden menos de 8 columnas, conviene subir `ventana.ancho_px` a 1024.

## Intensidad del render

Igual que en los experimentos previos (`render_espectrograma_uniforme` de
`espectrogramas_secuencias_comun.py`): STFT con los valores por defecto de librosa, dB respecto
al máximo del espectro completo (`amplitude_to_db(ref=np.max)`), recorte fijo a −80..0 dB, banda
20–100 kHz y colormap jet. No se aplica ninguna normalización ni filtro adicional.

Lo único que cambia respecto al pipeline previo es la geometría, que es necesaria para que todas
las pseudo-secuencias midan lo mismo: tasa común de 320 kHz y hop fijo (sin redimensionar), filas
sobre el Nyquist del audio en el piso en lugar de estirar, y las columnas que sobran tras el último
pulso en el piso (−80 dB).

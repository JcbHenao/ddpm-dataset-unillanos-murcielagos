# ============================================================
# FIGURA ILUSTRATIVA DEL DDPM
# Adaptada al modelo de espectrogramas de ecolocalización
#
# Muestra:
#
# 1. Proceso forward:
#       espectrograma real -> ruido progresivo
#
# 2. Proceso reverse:
#       ruido -> UNet2D -> eliminación progresiva
#             -> espectrograma sintético
#
# El proceso inverso se ejecuta REALMENTE con el modelo.
# ============================================================

import os
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from PIL import Image
from diffusers import DDPMPipeline


# ============================================================
# CONFIGURACIÓN
# ============================================================

# ------------------------------------------------------------
# MODELO REAL
# ------------------------------------------------------------

MODEL_DIR = (
    "/media/adalab/bodega/CamiloBuitrago/"
    "modelos-difusion-pruebas/"
    "ddpm-pulsos-molossus-128-1000-metrics/"
    "best_model"
)


# ------------------------------------------------------------
# DATASET
# ------------------------------------------------------------

DATASET_DIR = (
    "/media/adalab/bodega/CamiloBuitrago/"
    "espectrogramas/"
    "Muestra_Molossus_Alta_Calidad_128"
)


# ------------------------------------------------------------
# SALIDA
# ------------------------------------------------------------

OUTPUT_PATH = (
    "/media/adalab/bodega/CamiloBuitrago/"
    "modelos-difusion-pruebas/"
    "figura_ddpm_espectrogramas.png"
)


# ------------------------------------------------------------
# CONFIGURACIÓN EXPERIMENTAL
# ------------------------------------------------------------

IMAGE_SIZE = 128

SEED = 42

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# SEMILLAS
# ============================================================

torch.manual_seed(SEED)

np.random.seed(SEED)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(SEED)


# ============================================================
# CARGAR PIPELINE
# ============================================================

print("=" * 70)
print("CARGANDO MODELO DDPM")
print("=" * 70)

pipeline = DDPMPipeline.from_pretrained(
    MODEL_DIR
)

pipeline = pipeline.to(DEVICE)

unet = pipeline.unet

scheduler = pipeline.scheduler


# ------------------------------------------------------------
# INFORMACIÓN DEL MODELO
# ------------------------------------------------------------

NUM_TIMESTEPS = (
    scheduler.config.num_train_timesteps
)

MODEL_IMAGE_SIZE = (
    unet.config.sample_size
)

IN_CHANNELS = (
    unet.config.in_channels
)

OUT_CHANNELS = (
    unet.config.out_channels
)


print(
    f"Dispositivo       : {DEVICE}"
)

print(
    f"Timesteps         : {NUM_TIMESTEPS}"
)

print(
    f"Resolución        : "
    f"{MODEL_IMAGE_SIZE} × {MODEL_IMAGE_SIZE}"
)

print(
    f"Canales entrada    : {IN_CHANNELS}"
)

print(
    f"Canales salida     : {OUT_CHANNELS}"
)

print(
    "Arquitectura       : UNet2DModel"
)

print()


# ============================================================
# BUSCAR ESPECTROGRAMA REAL
# ============================================================

def buscar_imagen_real(dataset_dir):
    """
    Busca la primera imagen válida del conjunto.
    """

    extensiones = {
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".tif",
        ".tiff"
    }

    archivos = sorted(
        [
            p
            for p in Path(dataset_dir).rglob("*")
            if p.is_file()
            and p.suffix.lower() in extensiones
        ]
    )

    if not archivos:

        raise FileNotFoundError(
            "\nNo se encontraron imágenes en:\n"
            f"{dataset_dir}"
        )

    return archivos[0]


# ============================================================
# CARGAR ESPECTROGRAMA
# ============================================================

def cargar_imagen(
    path,
    size=IMAGE_SIZE
):
    """
    Carga el espectrograma y lo convierte al formato usado
    durante el entrenamiento del DDPM.

    RGB
    128 × 128
    [0,1] -> [-1,1]
    """

    image = Image.open(path).convert(
        "RGB"
    )

    image = image.resize(
        (size, size),
        Image.Resampling.BILINEAR
    )

    array = np.asarray(
        image,
        dtype=np.float32
    )

    array = array / 255.0

    tensor = torch.from_numpy(
        array
    )

    # H,W,C -> C,H,W
    tensor = tensor.permute(
        2,
        0,
        1
    )

    # [0,1] -> [-1,1]
    tensor = (
        tensor * 2.0
        - 1.0
    )

    return tensor.unsqueeze(0)


# ============================================================
# TENSOR -> IMAGEN
# ============================================================

def tensor_a_imagen(tensor):
    """
    Convierte tensor [-1,1] a imagen [0,1].
    """

    tensor = tensor.detach().cpu()

    tensor = tensor.squeeze(0)

    tensor = tensor.clamp(
        -1.0,
        1.0
    )

    tensor = (
        tensor + 1.0
    ) / 2.0

    tensor = tensor.permute(
        1,
        2,
        0
    )

    return tensor.numpy()


# ============================================================
# PROCESO FORWARD
# ============================================================

def generar_difusion_forward(
    x0,
    scheduler,
    timesteps
):
    """
    Aplica el proceso forward real del scheduler.

    x0 -> xt

    Se utiliza el mismo scheduler del modelo.
    """

    resultados = []

    for t in timesteps:

        t_tensor = torch.tensor(
            [t],
            device=x0.device,
            dtype=torch.long
        )

        # Para la figura utilizamos un ruido
        # reproducible para todas las etapas.
        ruido = torch.randn_like(
            x0
        )

        xt = scheduler.add_noise(
            x0,
            ruido,
            t_tensor
        )

        resultados.append(
            {
                "t": t,
                "image": xt
            }
        )

    return resultados


# ============================================================
# PROCESO REVERSE REAL
# ============================================================

def generar_difusion_reverse(
    unet,
    scheduler,
    seed,
    num_inference_steps
):
    """
    Ejecuta explícitamente el proceso inverso aprendido
    por el UNet2D.

    xT -> ... -> x0'
    """

    generator = torch.Generator(
        device=DEVICE
    ).manual_seed(seed)

    # --------------------------------------------------------
    # Configurar scheduler
    # --------------------------------------------------------

    scheduler.set_timesteps(
        num_inference_steps,
        device=DEVICE
    )

    timesteps = scheduler.timesteps

    # --------------------------------------------------------
    # Ruido inicial
    # --------------------------------------------------------

    image = torch.randn(
        (
            1,
            IN_CHANNELS,
            MODEL_IMAGE_SIZE,
            MODEL_IMAGE_SIZE
        ),
        generator=generator,
        device=DEVICE
    )

    # --------------------------------------------------------
    # Estados que vamos a conservar
    # --------------------------------------------------------

    indices_guardar = np.linspace(
        0,
        len(timesteps) - 1,
        5,
        dtype=int
    )

    resultados = []

    # --------------------------------------------------------
    # Denoising
    # --------------------------------------------------------

    with torch.no_grad():

        for step_index, timestep in enumerate(
            timesteps
        ):

            # -----------------------------------------------
            # UNet2D
            # -----------------------------------------------

            model_output = unet(
                image,
                timestep
            ).sample

            # -----------------------------------------------
            # Paso inverso del scheduler
            # -----------------------------------------------

            image = scheduler.step(
                model_output,
                timestep,
                image
            ).prev_sample

            # -----------------------------------------------
            # Guardar estado
            # -----------------------------------------------

            if step_index in indices_guardar:

                resultados.append(
                    {
                        "step": step_index,
                        "timestep": int(
                            timestep.item()
                            if torch.is_tensor(timestep)
                            else timestep
                        ),
                        "image": image.clone()
                    }
                )

    return resultados


# ============================================================
# CARGAR ESPECTROGRAMA REAL
# ============================================================

real_path = buscar_imagen_real(
    DATASET_DIR
)

print("=" * 70)
print("ESPECTROGRAMA DE REFERENCIA")
print("=" * 70)

print(
    real_path
)

print()


x0 = cargar_imagen(
    real_path
).to(DEVICE)


# ============================================================
# FORWARD
# ============================================================

# Cinco estados del proceso:
#
# t=0
# t=T/4
# t=T/2
# t=3T/4
# t=T

forward_timesteps = [
    0,
    NUM_TIMESTEPS // 4,
    NUM_TIMESTEPS // 2,
    (3 * NUM_TIMESTEPS) // 4,
    NUM_TIMESTEPS - 1
]


forward_results = generar_difusion_forward(
    x0,
    scheduler,
    forward_timesteps
)


# ============================================================
# REVERSE
# ============================================================

print("=" * 70)
print("EJECUTANDO PROCESO INVERSO DEL MODELO")
print("=" * 70)

reverse_results = generar_difusion_reverse(
    unet=unet,
    scheduler=scheduler,
    seed=SEED,
    num_inference_steps=NUM_TIMESTEPS
)

# ============================================================
# CREAR FIGURA
# ============================================================

fig = plt.figure(
    figsize=(18, 10),
    dpi=200
)

# ============================================================
# GRID
#
# 9 columnas:
#
# imagen | flecha | imagen | flecha | imagen | flecha | imagen | flecha | imagen
#
# Esto permite colocar las flechas EN LOS ESPACIOS ENTRE
# LOS ESPECTROGRAMAS y nunca encima de ellos.
# ============================================================

gs = fig.add_gridspec(
    nrows=5,
    ncols=9,

    # --------------------------------------------------------
    # Filas:
    #
    # 0 -> imágenes proceso forward
    # 1 -> descripción forward
    # 2 -> aprendizaje del proceso inverso
    # 3 -> imágenes proceso reverse
    # 4 -> descripción reverse
    # --------------------------------------------------------

    height_ratios=[
        1.0,
        0.22,
        0.28,
        1.0,
        0.30
    ],

    # --------------------------------------------------------
    # Columnas alternadas:
    #
    # imagen, flecha, imagen, flecha...
    # --------------------------------------------------------

    width_ratios=[
        1.0,
        0.12,
        1.0,
        0.12,
        1.0,
        0.12,
        1.0,
        0.12,
        1.0
    ],

    hspace=0.45,
    wspace=0.18
)


# ============================================================
# POSICIONES DE LAS CINCO IMÁGENES
#
# Columnas:
# 0 -> imagen
# 2 -> imagen
# 4 -> imagen
# 6 -> imagen
# 8 -> imagen
# ============================================================

columnas_imagen = [
    0,
    2,
    4,
    6,
    8
]


# ============================================================
# POSICIONES DE LAS FLECHAS
#
# Columnas:
# 1 -> entre imagen 1 y 2
# 3 -> entre imagen 2 y 3
# 5 -> entre imagen 3 y 4
# 7 -> entre imagen 4 y 5
# ============================================================

columnas_flecha = [
    1,
    3,
    5,
    7
]


# ============================================================
# ------------------------------------------------------------
# PROCESO FORWARD
# ------------------------------------------------------------
# ============================================================

for i, resultado in enumerate(
    forward_results
):

    columna = columnas_imagen[i]

    ax = fig.add_subplot(
        gs[0, columna]
    )

    ax.imshow(
        tensor_a_imagen(
            resultado["image"]
        )
    )

    t = resultado["t"]


    # --------------------------------------------------------
    # TÍTULOS
    # --------------------------------------------------------

    if i == 0:

        titulo = (
            "Espectrograma real\n"
            r"$x_0$"
        )

    elif i == len(
        forward_results
    ) - 1:

        titulo = (
            "Ruido gaussiano\n"
            r"$x_T$"
        )

    else:

        titulo = (
            "Ruido progresivo\n"
            f"t = {t}"
        )


    ax.set_title(
        titulo,
        fontsize=12,
        fontweight="bold",
        pad=10
    )

    ax.axis("off")


# ============================================================
# FLECHAS FORWARD
# ============================================================

for columna in columnas_flecha:

    ax_arrow = fig.add_subplot(
        gs[0, columna]
    )

    ax_arrow.axis("off")

    ax_arrow.text(
        0.5,
        0.5,
        "→",
        fontsize=28,
        ha="center",
        va="center",
        fontweight="bold"
    )


# ============================================================
# TEXTO DEL PROCESO FORWARD
#
# Ocupa prácticamente todo el ancho.
# ============================================================

ax_forward_text = fig.add_subplot(
    gs[1, :]
)

ax_forward_text.axis("off")

ax_forward_text.text(
    0.5,
    0.55,
    "PROCESO DE DIFUSIÓN HACIA ADELANTE",
    ha="center",
    va="center",
    fontsize=13,
    fontweight="bold"
)

ax_forward_text.text(
    0.5,
    0.08,
    "Se añade ruido gaussiano progresivamente "
    "hasta transformar el espectrograma en ruido.",
    ha="center",
    va="center",
    fontsize=11
)


# ============================================================
# SEPARADOR CENTRAL
#
# Aquí colocamos la explicación del aprendizaje.
# Ya no invade ninguna imagen.
# ============================================================

ax_middle = fig.add_subplot(
    gs[2, :]
)

ax_middle.axis("off")

ax_middle.text(
    0.5,
    0.72,
    "APRENDIZAJE DEL PROCESO INVERSO",
    ha="center",
    va="center",
    fontsize=12,
    fontweight="bold"
)

ax_middle.text(
    0.5,
    0.18,
    r"Durante el entrenamiento, la UNet2D aprende a estimar "
    r"el ruido $\epsilon_\theta(x_t,t)$ para cada paso temporal.",
    ha="center",
    va="center",
    fontsize=10.5
)


# ============================================================
# ------------------------------------------------------------
# PROCESO REVERSE
# ------------------------------------------------------------
# ============================================================

for i, resultado in enumerate(
    reverse_results
):

    columna = columnas_imagen[i]

    ax = fig.add_subplot(
        gs[3, columna]
    )

    ax.imshow(
        tensor_a_imagen(
            resultado["image"]
        )
    )

    timestep = resultado[
        "timestep"
    ]


    # --------------------------------------------------------
    # TÍTULOS
    # --------------------------------------------------------

    if i == 0:

        titulo = (
            "Ruido inicial\n"
            r"$x_T$"
        )

    elif i == len(
        reverse_results
    ) - 1:

        titulo = (
            "Espectrograma sintético\n"
            r"$x'_0$"
        )

    else:

        titulo = (
            "Eliminación de ruido\n"
            f"t = {timestep}"
        )


    ax.set_title(
        titulo,
        fontsize=12,
        fontweight="bold",
        pad=10
    )

    ax.axis("off")


# ============================================================
# FLECHAS REVERSE
# ============================================================

for columna in columnas_flecha:

    ax_arrow = fig.add_subplot(
        gs[3, columna]
    )

    ax_arrow.axis("off")

    ax_arrow.text(
        0.5,
        0.5,
        "→",
        fontsize=28,
        ha="center",
        va="center",
        fontweight="bold"
    )


# ============================================================
# TEXTO DEL PROCESO INVERSO
# ============================================================

ax_reverse_text = fig.add_subplot(
    gs[4, :]
)

ax_reverse_text.axis("off")

ax_reverse_text.text(
    0.5,
    0.72,
    "PROCESO INVERSO — GENERACIÓN",
    ha="center",
    va="center",
    fontsize=13,
    fontweight="bold"
)

ax_reverse_text.text(
    0.5,
    0.25,
    "A partir de ruido gaussiano, la UNet2D estima "
    "el ruido en cada paso y el scheduler actualiza "
    "progresivamente la muestra hasta obtener "
    "un espectrograma sintético.",
    ha="center",
    va="center",
    fontsize=10.5
)


# ============================================================
# INFORMACIÓN DEL MODELO
# ============================================================

fig.text(
    0.5,
    0.015,
    "DDPM no condicionado  |  UNet2D  |  "
    f"{NUM_TIMESTEPS} timesteps  |  "
    f"RGB {MODEL_IMAGE_SIZE}×{MODEL_IMAGE_SIZE}",
    ha="center",
    va="center",
    fontsize=10
)


# ============================================================
# AJUSTE FINAL
# ============================================================

plt.subplots_adjust(
    top=0.91,
    bottom=0.07,
    left=0.025,
    right=0.975
)


# ============================================================
# GUARDAR
# ============================================================

plt.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)


# ============================================================
# MOSTRAR
# ============================================================

plt.show()


# ============================================================
# INFORMACIÓN
# ============================================================

print()
print("=" * 70)
print("FIGURA GENERADA CORRECTAMENTE")
print("=" * 70)

print(
    f"Modelo:\n{MODEL_DIR}"
)

print(
    f"Imagen real:\n{real_path}"
)

print(
    f"Salida:\n{OUTPUT_PATH}"
)
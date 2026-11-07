import os
import json
import random
from pathlib import Path

import numpy as np
import torch

from PIL import Image

from torchvision import transforms

from tqdm.auto import tqdm

from diffusers import DDPMPipeline

from torchmetrics.image.fid import (
    FrechetInceptionDistance
)

from torchmetrics.image import (
    StructuralSimilarityIndexMeasure,
    InceptionScore
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

SEED = 42

IMAGE_WIDTH = 2048
IMAGE_HEIGHT = 128

# Número de imágenes generadas.
NUM_GENERATED = 200

# El conjunto test actual contiene 188 imágenes.
NUM_REAL = 188

# El modelo fue entrenado con 1500 timesteps.
NUM_INFERENCE_STEPS = 1500

# Para evitar un consumo excesivo de VRAM durante sampling.
GENERATION_BATCH_SIZE = 8

# Batch para FID e Inception Score.
METRIC_BATCH_SIZE = 8

# Batch de imágenes reales usado en cada comparación SSIM.
SSIM_REAL_BATCH_SIZE = 8


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# RUTAS
# ============================================================

SCRIPT_DIR = Path(
    __file__
).resolve().parent


BEST_MODEL_DIR = (
    SCRIPT_DIR
    / "ddpm-pulsos-saccopteryx-2048-1500ntt"
    / "best_model"
)


TEST_DIR = Path(
    "/media/adalab/bodega/CamiloBuitrago/"
    "espectrogramas/"
    "Espectrogramas_Secuencias_Chunks_1.5s_2048x128px/"
    "Dataset_DDPM_2048x128px/"
    "test"
)


OUTPUT_DIR = (
    SCRIPT_DIR
    / "evaluacion_best_ddpm"
    / "inference-1500"
)


GENERATED_DIR = (
    OUTPUT_DIR
    / "generated_1500"
)


METRICS_PATH = (
    OUTPUT_DIR
    / "metrics.json"
)


# ============================================================
# CREAR DIRECTORIOS
# ============================================================

GENERATED_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# REPRODUCIBILIDAD
# ============================================================

random.seed(
    SEED
)

np.random.seed(
    SEED
)

torch.manual_seed(
    SEED
)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(
        SEED
    )


# ============================================================
# INFORMACIÓN INICIAL
# ============================================================

print()
print("=" * 70)
print("EVALUACIÓN DEL BEST MODEL")
print("=" * 70)

print(
    f"Modelo               : "
    f"{BEST_MODEL_DIR}"
)

print(
    f"Test                 : "
    f"{TEST_DIR}"
)

print(
    f"Resolución           : "
    f"{IMAGE_WIDTH}x{IMAGE_HEIGHT}"
)

print(
    f"Timesteps entrenamiento : "
    f"1500"
)

print(
    f"Timesteps inferencia    : "
    f"{NUM_INFERENCE_STEPS}"
)

print(
    f"Imágenes reales       : "
    f"{NUM_REAL}"
)

print(
    f"Imágenes generadas    : "
    f"{NUM_GENERATED}"
)

print(
    f"Device               : "
    f"{DEVICE}"
)

print("=" * 70)


# ============================================================
# COMPROBAR RUTAS
# ============================================================

if not BEST_MODEL_DIR.exists():

    raise FileNotFoundError(
        "\nNo existe el best_model:\n"
        f"{BEST_MODEL_DIR}"
    )


if not TEST_DIR.exists():

    raise FileNotFoundError(
        "\nNo existe el conjunto test:\n"
        f"{TEST_DIR}"
    )


# ============================================================
# BUSCAR IMÁGENES REALES
# ============================================================

real_files = sorted(
    [
        path

        for path in TEST_DIR.rglob("*")

        if (
            path.is_file()
            and
            path.suffix.lower() == ".png"
        )
    ]
)


if len(real_files) == 0:

    raise ValueError(
        "\nNo se encontraron imágenes PNG en:\n"
        f"{TEST_DIR}"
    )


print()
print(
    f"PNG encontrados en test: "
    f"{len(real_files)}"
)


if len(real_files) < NUM_REAL:

    raise ValueError(
        "\nNo hay suficientes imágenes reales.\n"
        f"Disponibles: {len(real_files)}\n"
        f"Solicitadas: {NUM_REAL}"
    )


# ============================================================
# SELECCIÓN REPRODUCIBLE DE TEST
# ============================================================

selection_rng = random.Random(
    SEED
)


selected_real_files = selection_rng.sample(
    real_files,
    NUM_REAL
)


selected_real_files = sorted(
    selected_real_files
)


print(
    f"Imágenes reales utilizadas: "
    f"{len(selected_real_files)}"
)


# ============================================================
# TRANSFORMACIÓN
# ============================================================

to_tensor = transforms.ToTensor()


# ============================================================
# CARGAR IMÁGENES REALES
# ============================================================

def load_real_images(
    image_paths
):

    images = []

    for path in tqdm(
        image_paths,
        desc="Cargando reales"
    ):

        image = Image.open(
            path
        ).convert("RGB")


        # ----------------------------------------------------
        # NO REDIMENSIONAR
        # ----------------------------------------------------

        if image.size != (
            IMAGE_WIDTH,
            IMAGE_HEIGHT
        ):

            raise ValueError(
                "\nResolución incorrecta:\n"
                f"{path}\n"
                f"Esperado: "
                f"{IMAGE_WIDTH}x{IMAGE_HEIGHT}\n"
                f"Encontrado: "
                f"{image.size}"
            )


        tensor = to_tensor(
            image
        )


        images.append(
            tensor
        )


    return torch.stack(
        images
    )


real_images = load_real_images(
    selected_real_files
)


print()
print(
    f"Tensor reales: "
    f"{tuple(real_images.shape)}"
)

print(
    f"Dtype reales: "
    f"{real_images.dtype}"
)

print(
    f"Rango reales: "
    f"[{real_images.min().item():.3f}, "
    f"{real_images.max().item():.3f}]"
)


# ============================================================
# CARGAR BEST MODEL
# ============================================================

print()
print("=" * 70)
print("CARGANDO BEST MODEL")
print("=" * 70)


pipeline = DDPMPipeline.from_pretrained(
    str(BEST_MODEL_DIR)
)


pipeline = pipeline.to(
    DEVICE
)


pipeline.set_progress_bar_config(
    disable=True
)


pipeline.unet.eval()


print(
    "Best model cargado correctamente."
)


# ============================================================
# GENERACIÓN
# ============================================================

print()
print("=" * 70)
print("GENERANDO ESPECTROGRAMAS")
print("=" * 70)


generation_generator = torch.Generator(
    device=DEVICE
).manual_seed(
    SEED
)


generated_pil = []


generated_count = 0


with torch.inference_mode():

    progress = tqdm(
        total=NUM_GENERATED,
        desc="Generando"
    )


    while generated_count < NUM_GENERATED:

        current_batch_size = min(
            GENERATION_BATCH_SIZE,
            NUM_GENERATED - generated_count
        )


        images = pipeline(

            batch_size=current_batch_size,

            num_inference_steps=(
                NUM_INFERENCE_STEPS
            ),

            generator=generation_generator,

            output_type="pil"

        ).images


        for image in images:

            # ------------------------------------------------
            # COMPROBAR RESOLUCIÓN
            # ------------------------------------------------

            if image.size != (
                IMAGE_WIDTH,
                IMAGE_HEIGHT
            ):

                raise ValueError(
                    "\nEl modelo generó una "
                    "resolución inesperada:\n"
                    f"Encontrado: {image.size}\n"
                    f"Esperado: "
                    f"{IMAGE_WIDTH}x"
                    f"{IMAGE_HEIGHT}"
                )


            # ------------------------------------------------
            # GUARDAR EN MEMORIA
            # ------------------------------------------------

            generated_pil.append(
                image
            )


            # ------------------------------------------------
            # GUARDAR PNG
            # ------------------------------------------------

            output_path = (
                GENERATED_DIR
                /
                (
                    f"generated_"
                    f"{generated_count + 1:04d}.png"
                )
            )


            image.save(
                output_path
            )


            generated_count += 1


            progress.update(
                1
            )


    progress.close()


print()
print(
    f"Imágenes generadas: "
    f"{len(generated_pil)}"
)


# ============================================================
# CONVERTIR GENERADAS A TENSOR
# ============================================================

generated_images = torch.stack(
    [
        to_tensor(image)

        for image in generated_pil
    ]
)


print(
    f"Tensor generadas: "
    f"{tuple(generated_images.shape)}"
)

print(
    f"Dtype generadas: "
    f"{generated_images.dtype}"
)

print(
    f"Rango generadas: "
    f"[{generated_images.min().item():.3f}, "
    f"{generated_images.max().item():.3f}]"
)


# ============================================================
# COMPROBAR CANTIDADES
# ============================================================

if len(real_images) != NUM_REAL:

    raise ValueError(
        "La cantidad de imágenes reales "
        "no coincide con NUM_REAL."
    )


if len(generated_images) != NUM_GENERATED:

    raise ValueError(
        "La cantidad de imágenes generadas "
        "no coincide con NUM_GENERATED."
    )


# ============================================================
# FID
# ============================================================

print()
print("=" * 70)
print("CALCULANDO FID")
print("=" * 70)


fid = FrechetInceptionDistance(
    feature=2048,
    normalize=True
).to(
    DEVICE
)


# ------------------------------------------------------------
# REAL
# ------------------------------------------------------------

for start in tqdm(

    range(
        0,
        NUM_REAL,
        METRIC_BATCH_SIZE
    ),

    desc="FID reales"
):

    batch = real_images[
        start:
        start + METRIC_BATCH_SIZE
    ]


    fid.update(
        batch.to(DEVICE),
        real=True
    )


# ------------------------------------------------------------
# GENERATED
# ------------------------------------------------------------

for start in tqdm(

    range(
        0,
        NUM_GENERATED,
        METRIC_BATCH_SIZE
    ),

    desc="FID generadas"
):

    batch = generated_images[
        start:
        start + METRIC_BATCH_SIZE
    ]


    fid.update(
        batch.to(DEVICE),
        real=False
    )


fid_value = fid.compute().item()


print()
print(
    f"FID = {fid_value:.6f}"
)


# ============================================================
# INCEPTION SCORE
# ============================================================

print()
print("=" * 70)
print("CALCULANDO INCEPTION SCORE")
print("=" * 70)


inception_score = InceptionScore(
    normalize=True,
    splits=10
).to(
    DEVICE
)


for start in tqdm(

    range(
        0,
        NUM_GENERATED,
        METRIC_BATCH_SIZE
    ),

    desc="Inception Score"
):

    batch = generated_images[
        start:
        start + METRIC_BATCH_SIZE
    ]


    inception_score.update(
        batch.to(DEVICE)
    )


is_value = (
    inception_score.compute()[0].item()
)


print()
print(
    f"Inception Score = "
    f"{is_value:.6f}"
)


# ============================================================
# NEAREST SSIM
# ============================================================
#
# Para cada generada:
#
#       generated_i
#             │
#             ├── SSIM(real_1)
#             ├── SSIM(real_2)
#             ├── ...
#             └── SSIM(real_N)
#
# Se toma:
#
#       max_j SSIM(generated_i, real_j)
#
# Finalmente:
#
#       Nearest SSIM mean
#       =
#       promedio de todos los máximos
#
# ============================================================

print()
print("=" * 70)
print("CALCULANDO NEAREST SSIM")
print("=" * 70)


ssim_metric = StructuralSimilarityIndexMeasure(
    data_range=1.0,
    reduction="none"
).to(
    DEVICE
)


real_for_ssim = real_images.to(
    DEVICE
)


generated_for_ssim = generated_images.to(
    DEVICE
)


nearest_ssim_values = []


with torch.inference_mode():

    for generated_index in tqdm(

        range(
            NUM_GENERATED
        ),

        desc="Nearest SSIM"
    ):

        generated_image = (
            generated_for_ssim[
                generated_index
            ]
            .unsqueeze(0)
        )


        best_ssim = -1.0


        # ----------------------------------------------------
        # Comparación por bloques de reales
        # ----------------------------------------------------

        for start in range(

            0,

            NUM_REAL,

            SSIM_REAL_BATCH_SIZE
        ):

            real_batch = (
                real_for_ssim[
                    start:
                    start + SSIM_REAL_BATCH_SIZE
                ]
            )


            current_batch_size = (
                real_batch.shape[0]
            )


            generated_batch = (
                generated_image
                .expand(
                    current_batch_size,
                    -1,
                    -1,
                    -1
                )
            )


            scores = ssim_metric(

                generated_batch,

                real_batch

            )


            current_best = (
                scores.max().item()
            )


            if current_best > best_ssim:

                best_ssim = (
                    current_best
                )


        nearest_ssim_values.append(
            best_ssim
        )


nearest_ssim_mean = float(
    np.mean(
        nearest_ssim_values
    )
)


print()
print(
    f"Nearest SSIM mean = "
    f"{nearest_ssim_mean:.6f}"
)


# ============================================================
# RESULTADOS
# ============================================================

results = {

    "fid": fid_value,

    "nearest_ssim_mean": (
        nearest_ssim_mean
    ),

    "inception_score": (
        is_value
    )
}


# ============================================================
# GUARDAR JSON
# ============================================================

with open(

    METRICS_PATH,

    "w",

    encoding="utf-8"

) as file:

    json.dump(

        results,

        file,

        indent=4,

        ensure_ascii=False
    )


# ============================================================
# RESULTADO FINAL
# ============================================================

print()
print("=" * 70)
print("RESULTADOS FINALES")
print("=" * 70)

print(
    f"FID                 : "
    f"{fid_value:.6f}"
)

print(
    f"Nearest SSIM mean   : "
    f"{nearest_ssim_mean:.6f}"
)

print(
    f"Inception Score     : "
    f"{is_value:.6f}"
)

print()
print(
    f"Muestras generadas:"
    f"\n{GENERATED_DIR}"
)

print()
print(
    f"Métricas:"
    f"\n{METRICS_PATH}"
)

print("=" * 70)
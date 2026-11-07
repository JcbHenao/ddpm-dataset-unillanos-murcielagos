import os
import csv
import json
import random

import numpy as np
import torch

from PIL import Image
from scipy.stats import ks_2samp

from torchvision import transforms

from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image import StructuralSimilarityIndexMeasure

from diffusers import DDPMPipeline


# ============================================================
# CONFIGURACIÓN
# ============================================================

SEED = 42

IMAGE_SIZE = 128

NUM_REAL = 200
NUM_GENERATED = 200

INFERENCE_STEPS = 500


DATASET_DIR = (
    "/media/adalab/bodega/CamiloBuitrago/"
    "espectrogramas/"
    "Muestra_Molossus_Alta_Calidad_128"
)


EXPERIMENT_ROOT = (
    "ddpm-grid-experimentos"
)


EVALUATION_ROOT = os.path.join(
    EXPERIMENT_ROOT,
    "evaluation"
)

EXPERIMENTS = [

    # ========================================================
    # 500 TIMESTEPS
    # ========================================================

    {
        "name": "ts500_lr2.5e5",
        "timesteps": 500,
        "learning_rate": 2.5e-5,
    },

    {
        "name": "ts500_lr5e5",
        "timesteps": 500,
        "learning_rate": 5e-5,
    },

    {
        "name": "ts500_lr1e4",
        "timesteps": 500,
        "learning_rate": 1e-4,
    },

    {
        "name": "ts500_lr2e4",
        "timesteps": 500,
        "learning_rate": 2e-4,
    },


    # ========================================================
    # 1000 TIMESTEPS
    # ========================================================

    {
        "name": "ts1000_lr2.5e5",
        "timesteps": 1000,
        "learning_rate": 2.5e-5,
    },

    {
        "name": "ts1000_lr5e5",
        "timesteps": 1000,
        "learning_rate": 5e-5,
    },

    {
        "name": "ts1000_lr1e4",
        "timesteps": 1000,
        "learning_rate": 1e-4,
    },

    {
        "name": "ts1000_lr2e4",
        "timesteps": 1000,
        "learning_rate": 2e-4,
    },


    # ========================================================
    # 1500 TIMESTEPS
    # ========================================================

    {
        "name": "ts1500_lr2.5e5",
        "timesteps": 1500,
        "learning_rate": 2.5e-5,
    },

    {
        "name": "ts1500_lr5e5",
        "timesteps": 1500,
        "learning_rate": 5e-5,
    },

    {
        "name": "ts1500_lr1e4",
        "timesteps": 1500,
        "learning_rate": 1e-4,
    },

    {
        "name": "ts1500_lr2e4",
        "timesteps": 1500,
        "learning_rate": 2e-4,
    },

]



# ============================================================
# SEMILLAS
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(
        SEED
    )


DEVICE = torch.device(

    "cuda"
    if torch.cuda.is_available()
    else "cpu"

)


# ============================================================
# DATASET
# ============================================================

class EspectrogramasDataset:

    def __init__(
        self,
        carpeta
    ):

        self.carpeta = carpeta

        self.archivos = sorted([

            f

            for f in os.listdir(carpeta)

            if f.lower().endswith(".png")

        ])


        if len(self.archivos) == 0:

            raise ValueError(

                f"No se encontraron imágenes PNG en:\n"
                f"{carpeta}"

            )


        self.transform = transforms.Compose([

            transforms.Resize(

                (
                    IMAGE_SIZE,
                    IMAGE_SIZE
                )

            ),

            transforms.ToTensor(),

        ])


    def __len__(self):

        return len(
            self.archivos
        )


    def __getitem__(
        self,
        idx
    ):

        ruta = os.path.join(

            self.carpeta,

            self.archivos[idx]

        )


        image = Image.open(
            ruta
        ).convert("RGB")


        return self.transform(
            image
        )


# ============================================================
# DATASET Y SPLIT
# ============================================================

dataset = EspectrogramasDataset(
    DATASET_DIR
)


if len(dataset) != 1000:

    raise ValueError(

        f"Se esperaban 1000 imágenes, "
        f"pero se encontraron {len(dataset)}."

    )


# ------------------------------------------------------------
# IMPORTANTE:
#
# El split debe coincidir exactamente con el entrenamiento.
#
# ------------------------------------------------------------

from torch.utils.data import random_split


split_generator = (
    torch.Generator()
    .manual_seed(SEED)
)


train_dataset, val_dataset, test_dataset = random_split(

    dataset,

    [
        600,
        200,
        200
    ],

    generator=split_generator

)


print()
print("=" * 80)
print("DATASET")
print("=" * 80)

print(
    f"Total      : {len(dataset)}"
)

print(
    f"Train      : {len(train_dataset)}"
)

print(
    f"Validation : {len(val_dataset)}"
)

print(
    f"Test       : {len(test_dataset)}"
)


# ============================================================
# IMÁGENES REALES
# ============================================================

real_images = torch.stack([

    test_dataset[i]

    for i in range(
        len(test_dataset)
    )

])


if real_images.shape[0] != NUM_REAL:

    raise ValueError(

        "El conjunto de prueba no tiene "
        f"{NUM_REAL} imágenes."

    )


print(
    "Real:",
    real_images.shape
)


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def guardar_json(
    datos,
    ruta
):

    with open(

        ruta,

        "w",

        encoding="utf-8"

    ) as f:

        json.dump(

            datos,

            f,

            indent=4

        )


# ============================================================
# GENERACIÓN
# ============================================================

def generar_muestras(

    model_dir,

    generated_dir

):

    os.makedirs(

        generated_dir,

        exist_ok=True

    )


    print()
    print(
        "Cargando modelo:"
    )

    print(
        model_dir
    )


    pipeline = (
        DDPMPipeline
        .from_pretrained(
            model_dir
        )
        .to(
            DEVICE
        )
    )


    pipeline.set_progress_bar_config(

        disable=False

    )


    generator = (

        torch.Generator(
            device=DEVICE
        )
        .manual_seed(SEED)

    )


    generated_images = []


    remaining = NUM_GENERATED


    # --------------------------------------------------------
    # Generación por lotes para no consumir toda la VRAM
    # --------------------------------------------------------

    while remaining > 0:

        batch_size = min(

            16,

            remaining

        )


        print(

            f"Generando {batch_size} "
            f"imágenes..."

        )


        with torch.no_grad():

            batch_images = pipeline(

                batch_size=batch_size,

                num_inference_steps=
                    INFERENCE_STEPS,

                generator=generator

            ).images


        generated_images.extend(
            batch_images
        )


        remaining -= batch_size


    # --------------------------------------------------------
    # Guardar imágenes
    # --------------------------------------------------------

    for i, image in enumerate(
        generated_images
    ):

        path = os.path.join(

            generated_dir,

            f"generated_{i:04d}.png"

        )


        image.save(
            path
        )


    print(
        f"Generadas: "
        f"{len(generated_images)}"
    )


    # Liberar pipeline después de generar
    del pipeline

    torch.cuda.empty_cache()


    return generated_images


# ============================================================
# CONVERTIR SINTÉTICAS
# ============================================================

def convertir_a_tensor(
    images
):

    tensors = [

        transforms.ToTensor()(
            image
        )

        for image in images

    ]


    return torch.stack(
        tensors
    )


# ============================================================
# FID
# ============================================================

def calcular_fid(

    real,

    synthetic

):

    print()
    print("Calculando FID...")
    
    metric = FrechetInceptionDistance(feature=2048,normalize=True).to(DEVICE)
    real = real.to(DEVICE)
    synthetic = synthetic.to(DEVICE)


    with torch.no_grad():

        metric.update(real,real=True)
        metric.update(synthetic,real=False)
    value = metric.compute().item()


    del metric

    torch.cuda.empty_cache()


    return value


# ============================================================
# NEAREST SSIM
# ============================================================
#
# Para cada conjunto:
#
#   REAL:
#       cada R_i contra todos los R_j, i != j
#
#   SINTÉTICO:
#       cada G_i contra todos los G_j, i != j
#
# Se obtiene una distribución de nearest SSIM para cada conjunto.
#
# ============================================================

def calcular_distribucion_nearest_ssim(images,nombre):

    print()
    print(
        f"Calculando nearest SSIM: "
        f"{nombre}"
    )


    metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)


    images = images.to(
        DEVICE
    )


    n = images.shape[0]


    nearest_values = []


    CHUNK_SIZE = 32


    with torch.no_grad():

        for i in range(n):

            reference = (images[i].unsqueeze(0))

            best_ssim = -1.0

            # ------------------------------------------------
            # Primera parte: imágenes antes de i
            # ------------------------------------------------

            for start in range(0,i,CHUNK_SIZE):

                end = min(

                    start + CHUNK_SIZE,

                    i

                )


                candidates = images[
                    start:end
                ]


                if candidates.shape[0] == 0:

                    continue


                reference_batch = (
                    reference
                    .expand(
                        candidates.shape[0],
                        -1,
                        -1,
                        -1
                    )
                )


                values = metric(

                    reference_batch,

                    candidates

                )

                if values.ndim == 0:

                    # Fallback seguro
                    values_list = []

                    for k in range(
                        candidates.shape[0]
                    ):

                        value = metric(

                            reference,

                            candidates[k:k+1]

                        ).item()

                        values_list.append(
                            value
                        )

                    chunk_max = max(
                        values_list
                    )

                else:

                    chunk_max = (
                        values.max().item()
                    )


                best_ssim = max(

                    best_ssim,

                    chunk_max

                )


            # ------------------------------------------------
            # Segunda parte: imágenes después de i
            # ------------------------------------------------

            for start in range(i + 1,n,CHUNK_SIZE):

                end = min(start + CHUNK_SIZE,n)

                candidates = images[start:end]

                if candidates.shape[0] == 0:
                    continue


                reference_batch = (
                    reference
                    .expand(
                        candidates.shape[0],
                        -1,
                        -1,
                        -1
                    )
                )


                values = metric(

                    reference_batch,

                    candidates

                )


                if values.ndim == 0:

                    values_list = []

                    for k in range(
                        candidates.shape[0]
                    ):

                        value = metric(

                            reference,

                            candidates[k:k+1]

                        ).item()

                        values_list.append(
                            value
                        )

                    chunk_max = max(
                        values_list
                    )

                else:

                    chunk_max = (
                        values.max().item()
                    )


                best_ssim = max(

                    best_ssim,

                    chunk_max

                )


            nearest_values.append(
                best_ssim
            )


            if (
                (i + 1) % 25 == 0
            ):

                print(

                    f"{nombre}: "
                    f"{i + 1}/{n}"

                )


    del metric
    torch.cuda.empty_cache()


    values = np.asarray(

        nearest_values,

        dtype=np.float64

    )


    return {

        "mean":
            float(values.mean()),

        "std":
            float(values.std()),

        "min":
            float(values.min()),

        "max":
            float(values.max()),

        "values":
            values

    }


# ============================================================
# COMPARAR DISTRIBUCIONES
# ============================================================
#
# Real nearest SSIM
#             VS
# Synthetic nearest SSIM
#
# Se utiliza KS como medida descriptiva de diferencia entre
# distribuciones.
#
# ============================================================

real_ssim = (calcular_distribucion_nearest_ssim(real_images,"REAL"))

def comparar_distribuciones(

    real_values,

    synthetic_values

):

    ks_statistic, ks_pvalue = ks_2samp(

        real_values,

        synthetic_values

    )


    return {

        "ks_statistic":
            float(ks_statistic),

        "ks_pvalue":
            float(ks_pvalue)

    }


# ============================================================
# EVALUAR UNA CONFIGURACIÓN
# ============================================================

def evaluar_experimento(

    experiment

):

    name = experiment["name"]


    model_dir = os.path.join(

        EXPERIMENT_ROOT,

        name,

        "best_model"

    )


    if not os.path.isdir(
        model_dir
    ):

        raise FileNotFoundError(

            f"No existe:\n"
            f"{model_dir}"

        )


    output_dir = os.path.join(

        EVALUATION_ROOT,

        name

    )


    generated_dir = os.path.join(

        output_dir,

        "generated"

    )


    metrics_dir = os.path.join(

        output_dir,

        "metrics"

    )


    os.makedirs(

        metrics_dir,

        exist_ok=True

    )


    # ========================================================
    # GENERAR
    # ========================================================

    generated_images = generar_muestras(

        model_dir,

        generated_dir

    )


    generated = convertir_a_tensor(

        generated_images

    )


    print()
    print("=" * 80)

    print(
        f"EVALUANDO: {name}"
    )

    print("=" * 80)

    print(
        f"Timesteps: "
        f"{experiment['timesteps']}"
    )

    print(
        f"Learning rate: "
        f"{experiment['learning_rate']}"
    )

    print(
        f"Real: {real_images.shape}"
    )

    print(
        f"Synthetic: {generated.shape}"
    )


    # ========================================================
    # FID
    # ========================================================

    fid = calcular_fid(

        real_images,

        generated

    )



    # ========================================================
    # NEAREST SSIM SINTÉTICO
    # ========================================================

    synthetic_ssim = (
        calcular_distribucion_nearest_ssim(

            generated,

            "SINTÉTICO"

        )
    )


    # ========================================================
    # COMPARACIÓN
    # ========================================================

    distribution_difference = (
        comparar_distribuciones(

            real_ssim["values"],

            synthetic_ssim["values"]

        )
    )


    # ========================================================
    # RESULTADOS
    # ========================================================

    results = {

        "experiment":
            name,

        "timesteps":
            experiment["timesteps"],

        "learning_rate":
            experiment["learning_rate"],

        "seed":
            SEED,

        "num_real":
            NUM_REAL,

        "num_generated":
            NUM_GENERATED,

        "image_size":
            IMAGE_SIZE,

        "inference_steps":
            INFERENCE_STEPS,

        # ---------------- FID ----------------

        "fid":
            fid,

        # ---------- Nearest SSIM real ----------

        "real_nearest_ssim_mean":
            real_ssim["mean"],

        "real_nearest_ssim_std":
            real_ssim["std"],

        "real_nearest_ssim_min":
            real_ssim["min"],

        "real_nearest_ssim_max":
            real_ssim["max"],

        # ------- Nearest SSIM synthetic -------

        "synthetic_nearest_ssim_mean":
            synthetic_ssim["mean"],

        "synthetic_nearest_ssim_std":
            synthetic_ssim["std"],

        "synthetic_nearest_ssim_min":
            synthetic_ssim["min"],

        "synthetic_nearest_ssim_max":
            synthetic_ssim["max"],

        # -------- Diferencia distribución ------

        "ks_statistic":
            distribution_difference[
                "ks_statistic"
            ],

        "ks_pvalue":
            distribution_difference[
                "ks_pvalue"
            ],

    }


    # ========================================================
    # GUARDAR JSON
    # ========================================================

    json_path = os.path.join(

        metrics_dir,

        "generative_metrics.json"

    )


    # No guardar los arrays de SSIM en JSON
    json_data = dict(results)


    guardar_json(

        json_data,

        json_path

    )


    # ========================================================
    # CSV INDIVIDUAL
    # ========================================================

    csv_path = os.path.join(

        metrics_dir,

        "generative_metrics.csv"

    )


    with open(

        csv_path,

        "w",

        newline="",

        encoding="utf-8"

    ) as f:

        writer = csv.DictWriter(

            f,

            fieldnames=results.keys()

        )

        writer.writeheader()

        writer.writerow(
            results
        )


    # ========================================================
    # GUARDAR DISTRIBUCIONES
    # ========================================================

    distribution_path = os.path.join(

        metrics_dir,

        "nearest_ssim_distributions.csv"

    )


    with open(

        distribution_path,

        "w",

        newline="",

        encoding="utf-8"

    ) as f:

        writer = csv.writer(
            f
        )

        writer.writerow([

            "index",

            "real_nearest_ssim",

            "synthetic_nearest_ssim"

        ])


        for i in range(
            NUM_REAL
        ):

            writer.writerow([

                i,

                real_ssim[
                    "values"
                ][i],

                synthetic_ssim[
                    "values"
                ][i],

            ])


    # ========================================================
    # RESULTADO EN CONSOLA
    # ========================================================

    print()
    print(
        "=" * 80
    )

    print(
        f"RESULTADOS: {name}"
    )

    print(
        f"FID: "
        f"{fid:.6f}"
    )

    print()

    print(
        "Nearest SSIM REAL"
    )

    print(
        f"  Media: "
        f"{real_ssim['mean']:.6f}"
    )

    print(
        f"  Std: "
        f"{real_ssim['std']:.6f}"
    )

    print(
        f"  Min: "
        f"{real_ssim['min']:.6f}"
    )

    print(
        f"  Max: "
        f"{real_ssim['max']:.6f}"
    )

    print()

    print(
        "Nearest SSIM SINTÉTICO"
    )

    print(
        f"  Media: "
        f"{synthetic_ssim['mean']:.6f}"
    )

    print(
        f"  Std: "
        f"{synthetic_ssim['std']:.6f}"
    )

    print(
        f"  Min: "
        f"{synthetic_ssim['min']:.6f}"
    )

    print(
        f"  Max: "
        f"{synthetic_ssim['max']:.6f}"
    )

    print()

    print(
        "DIFERENCIA ENTRE DISTRIBUCIONES"
    )

    print(
        f"  KS statistic: "
        f"{distribution_difference['ks_statistic']:.6f}"
    )

    print(
        f"  KS p-value: "
        f"{distribution_difference['ks_pvalue']:.6f}"
    )

    print()

    print(
        f"Guardado en: "
        f"{metrics_dir}"
    )

    print("=" * 80)


    # Liberar memoria
    del generated

    del generated_images

    torch.cuda.empty_cache()


    return results


# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs(

        EVALUATION_ROOT,

        exist_ok=True

    )


    all_results = []


    # ========================================================
    # RECORRER GRID
    # ========================================================

    for index, experiment in enumerate(

        EXPERIMENTS,

        start=1

    ):

        print()
        print("#" * 80)

        print(

            f"EXPERIMENTO {index}/"
            f"{len(EXPERIMENTS)}"

        )

        print(
            experiment["name"]
        )

        print("#" * 80)


        result = evaluar_experimento(
            experiment
        )


        all_results.append(
            result
        )


    # ========================================================
    # SUMMARY
    # ========================================================

    summary_path = os.path.join(

        EVALUATION_ROOT,

        "evaluation_summary.csv"

    )


    if all_results:

        with open(

            summary_path,

            "w",

            newline="",

            encoding="utf-8"

        ) as f:

            writer = csv.DictWriter(

                f,

                fieldnames=all_results[0].keys()

            )

            writer.writeheader()

            writer.writerows(
                all_results
            )


    # ========================================================
    # MOSTRAR RESUMEN
    # ========================================================

    print()
    print("=" * 100)
    print("RESUMEN FINAL")
    print("=" * 100)

    for result in all_results:

        print()

        print(
            result["experiment"]
        )

        print(
            f"  Timesteps: "
            f"{result['timesteps']}"
        )

        print(
            f"  Learning rate: "
            f"{result['learning_rate']}"
        )

        print(
            f"  FID: "
            f"{result['fid']:.6f}"
        )

        print(
            f"  Real nearest SSIM: "
            f"{result['real_nearest_ssim_mean']:.6f}"
        )

        print(
            f"  Synthetic nearest SSIM: "
            f"{result['synthetic_nearest_ssim_mean']:.6f}"
        )

        print(
            f"  KS statistic: "
            f"{result['ks_statistic']:.6f}"
        )


    print()
    print(
        "Resumen guardado en:"
    )

    print(
        summary_path
    )

    print("=" * 100)


# ============================================================
# EJECUCIÓN
# ============================================================

if __name__ == "__main__":

    main()

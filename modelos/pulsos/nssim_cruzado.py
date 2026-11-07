"""
============================================================
EXPERIMENTO SEPARADO: NEAREST SSIM CRUZADO (sintético -> real)
============================================================

Objetivo
--------
Determinar con qué combinación de (timesteps, learning rate) el DDPM
genera espectrogramas MÁS CERCANOS al conjunto real.

Diferencia con el script de evaluación anterior
-----------------------------------------------
El script anterior calculaba el nearest SSIM DENTRO de cada conjunto
(real vs real, sintético vs sintético) y comparaba ambas distribuciones
con un KS. Eso mide la diversidad interna de cada conjunto, no la
cercanía entre ellos.

Aquí se calcula el nearest SSIM CRUZADO, que es el definido en el
artículo:

    nSSIM_i = max_j SSIM(G_i, R_j)

es decir, para cada imagen generada G_i se busca la imagen real R_j
más parecida y se guarda ese valor. La métrica de resumen por
configuración es la MEDIA de esos máximos.

Se calcula contra DOS referencias, y ambas importan:

  * TEST  (200 imágenes no vistas)  -> fidelidad.
        Media alta = las muestras se parecen a pulsos reales que el
        modelo nunca vio. Es la métrica para rankear configuraciones.

  * TRAIN (600 imágenes vistas)     -> memorización.
        Si nSSIM_train es claramente mayor que nSSIM_test, o si hay
        muchas muestras con nSSIM_train > 0.90, el modelo está
        copiando ejemplos de entrenamiento en lugar de generalizar.
        Una media alta por memorización NO es un buen resultado.

Como línea base se calcula también el nearest SSIM de REAL_TEST contra
REAL_TRAIN: es el techo razonable al que puede aspirar el modelo, ya que
mide cuánto se parecen entre sí dos pulsos reales distintos.

Salidas
-------
  ddpm-grid-experimentos/evaluation_nssim/nssim_summary.csv
  ddpm-grid-experimentos/evaluation_nssim/nssim_tabla.tex
  ddpm-grid-experimentos/evaluation_nssim/<config>/nssim_values.csv

============================================================
"""

import os
import csv
import json
import random

import numpy as np
import torch

from PIL import Image

from torchvision import transforms
from torch.utils.data import random_split

from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.functional.image import (
    structural_similarity_index_measure as ssim_fn
)

from diffusers import DDPMPipeline


# ============================================================
# CONFIGURACIÓN
# ============================================================

SEED = 42

IMAGE_SIZE = 128

NUM_GENERATED = 200

# Pasos de muestreo en inferencia.
# None = usar los mismos timesteps con que se entrenó cada modelo
# (recomendado: comparar 500 vs 1000 con un valor fijo de 500 pasos
#  de inferencia mezcla dos efectos distintos).
INFERENCE_STEPS = None

# Umbral por encima del cual se considera que una muestra es
# prácticamente una copia de una imagen real.
DUPLICADO_UMBRAL = 0.90

CHUNK_SIZE = 50


DATASET_DIR = (
    "/media/adalab/bodega/CamiloBuitrago/"
    "espectrogramas/"
    "Muestra_Molossus_Alta_Calidad_128"
)

EXPERIMENT_ROOT = "ddpm-grid-experimentos"

OUTPUT_ROOT = os.path.join(
    EXPERIMENT_ROOT,
    "evaluation_nssim"
)

# Si ya existen imágenes generadas por el script de evaluación previo,
# se reutilizan en lugar de volver a muestrear (ahorra horas de GPU).
REUSE_GENERATED_FROM = os.path.join(
    EXPERIMENT_ROOT,
    "evaluation"
)


EXPERIMENTS = [
    {"name": "ts500_lr2.5e5",  "timesteps": 500,  "learning_rate": 2.5e-5},
    {"name": "ts500_lr5e5",    "timesteps": 500,  "learning_rate": 5e-5},
    {"name": "ts500_lr1e4",    "timesteps": 500,  "learning_rate": 1e-4},
    {"name": "ts500_lr2e4",    "timesteps": 500,  "learning_rate": 2e-4},

    {"name": "ts1000_lr2.5e5", "timesteps": 1000, "learning_rate": 2.5e-5},
    {"name": "ts1000_lr5e5",   "timesteps": 1000, "learning_rate": 5e-5},
    {"name": "ts1000_lr1e4",   "timesteps": 1000, "learning_rate": 1e-4},
    {"name": "ts1000_lr2e4",   "timesteps": 1000, "learning_rate": 2e-4},

    {"name": "ts1500_lr2.5e5", "timesteps": 1500, "learning_rate": 2.5e-5},
    {"name": "ts1500_lr5e5",   "timesteps": 1500, "learning_rate": 5e-5},
    {"name": "ts1500_lr1e4",   "timesteps": 1500, "learning_rate": 1e-4},
    {"name": "ts1500_lr2e4",   "timesteps": 1500, "learning_rate": 2e-4},
]


# ============================================================
# SEMILLAS Y DISPOSITIVO
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# DATASET (idéntico al del entrenamiento)
# ============================================================

class EspectrogramasDataset:

    def __init__(self, carpeta):

        self.carpeta = carpeta

        self.archivos = sorted([
            f
            for f in os.listdir(carpeta)
            if f.lower().endswith(".png")
        ])

        if len(self.archivos) == 0:
            raise ValueError(
                f"No se encontraron imágenes PNG en:\n{carpeta}"
            )

        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.archivos)

    def __getitem__(self, idx):

        ruta = os.path.join(
            self.carpeta,
            self.archivos[idx]
        )

        image = Image.open(ruta).convert("RGB")

        return self.transform(image)


def cargar_conjuntos():

    dataset = EspectrogramasDataset(DATASET_DIR)

    if len(dataset) != 1000:
        raise ValueError(
            f"Se esperaban 1000 imágenes, "
            f"pero se encontraron {len(dataset)}."
        )

    # MISMA semilla y MISMO orden que en el entrenamiento:
    # de lo contrario el "test" contendría imágenes vistas.
    split_generator = torch.Generator().manual_seed(SEED)

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [600, 200, 200],
        generator=split_generator
    )

    train = torch.stack([
        train_dataset[i] for i in range(len(train_dataset))
    ])

    test = torch.stack([
        test_dataset[i] for i in range(len(test_dataset))
    ])

    return train, test


# ============================================================
# NEAREST SSIM CRUZADO
# ============================================================

@torch.no_grad()
def nearest_ssim_cruzado(consulta, referencia, nombre=""):
    """
    Para cada imagen de `consulta` devuelve el SSIM máximo
    contra todas las imágenes de `referencia`, junto con el
    índice de la referencia más parecida.

    consulta   : (N, 3, H, W) en [0, 1]
    referencia : (M, 3, H, W) en [0, 1]
    """

    consulta = consulta.to(DEVICE)
    referencia = referencia.to(DEVICE)

    n = consulta.shape[0]
    m = referencia.shape[0]

    mejores = np.zeros(n, dtype=np.float64)
    indices = np.zeros(n, dtype=np.int64)

    for i in range(n):

        actual = consulta[i].unsqueeze(0)

        mejor_valor = -1.0
        mejor_indice = -1

        for start in range(0, m, CHUNK_SIZE):

            end = min(start + CHUNK_SIZE, m)

            candidatos = referencia[start:end]

            lote = actual.expand(
                candidatos.shape[0], -1, -1, -1
            )

            # reduction="none" -> un valor de SSIM por par, no el promedio del lote.
            valores = ssim_fn(
                lote,
                candidatos,
                data_range=1.0,
                reduction="none"
            )

            valores = valores.reshape(-1)

            valor_max, pos = torch.max(valores, dim=0)

            valor_max = valor_max.item()

            if valor_max > mejor_valor:
                mejor_valor = valor_max
                mejor_indice = start + pos.item()

        mejores[i] = mejor_valor
        indices[i] = mejor_indice

        if nombre and (i + 1) % 50 == 0:
            print(f"  {nombre}: {i + 1}/{n}")

    torch.cuda.empty_cache()

    return mejores, indices


def resumen(valores):

    return {
        "mean": float(np.mean(valores)),
        "std": float(np.std(valores)),
        "median": float(np.median(valores)),
        "min": float(np.min(valores)),
        "max": float(np.max(valores)),
        "pct_sobre_umbral": float(
            100.0 * np.mean(valores > DUPLICADO_UMBRAL)
        ),
    }


# ============================================================
# FID
# ============================================================

@torch.no_grad()
def calcular_fid(real, sintetico):

    metric = FrechetInceptionDistance(
        feature=2048,
        normalize=True
    ).to(DEVICE)

    metric.update(real.to(DEVICE), real=True)
    metric.update(sintetico.to(DEVICE), real=False)

    valor = metric.compute().item()

    del metric
    torch.cuda.empty_cache()

    return valor


# ============================================================
# GENERACIÓN / CARGA DE MUESTRAS SINTÉTICAS
# ============================================================

def cargar_png_como_tensor(carpeta):

    archivos = sorted([
        f for f in os.listdir(carpeta)
        if f.lower().endswith(".png")
    ])

    to_tensor = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    tensores = [
        to_tensor(
            Image.open(
                os.path.join(carpeta, f)
            ).convert("RGB")
        )
        for f in archivos
    ]

    return torch.stack(tensores)


def obtener_sinteticas(experiment, generated_dir):

    name = experiment["name"]

    # --------------------------------------------------------
    # 1. Reutilizar muestras ya generadas
    # --------------------------------------------------------

    previo = os.path.join(
        REUSE_GENERATED_FROM,
        name,
        "generated"
    )

    for carpeta in (generated_dir, previo):

        if os.path.isdir(carpeta):

            existentes = [
                f for f in os.listdir(carpeta)
                if f.lower().endswith(".png")
            ]

            if len(existentes) >= NUM_GENERATED:

                print(
                    f"Reutilizando {len(existentes)} muestras de:\n"
                    f"  {carpeta}"
                )

                tensor = cargar_png_como_tensor(carpeta)

                return tensor[:NUM_GENERATED]

    # --------------------------------------------------------
    # 2. Generar desde el best_model
    # --------------------------------------------------------

    model_dir = os.path.join(
        EXPERIMENT_ROOT,
        name,
        "best_model"
    )

    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"No existe:\n{model_dir}")

    os.makedirs(generated_dir, exist_ok=True)

    pipeline = DDPMPipeline.from_pretrained(model_dir).to(DEVICE)

    pasos = (
        INFERENCE_STEPS
        if INFERENCE_STEPS is not None
        else pipeline.scheduler.config.num_train_timesteps
    )

    print(f"Generando {NUM_GENERATED} muestras con {pasos} pasos...")

    generator = torch.Generator(device=DEVICE).manual_seed(SEED)

    imagenes = []
    restantes = NUM_GENERATED

    while restantes > 0:

        batch_size = min(16, restantes)

        with torch.no_grad():
            lote = pipeline(
                batch_size=batch_size,
                num_inference_steps=pasos,
                generator=generator
            ).images

        imagenes.extend(lote)
        restantes -= batch_size

    for i, imagen in enumerate(imagenes):
        imagen.save(
            os.path.join(generated_dir, f"generated_{i:04d}.png")
        )

    del pipeline
    torch.cuda.empty_cache()

    to_tensor = transforms.ToTensor()

    return torch.stack([to_tensor(img) for img in imagenes])


# ============================================================
# EVALUAR UNA CONFIGURACIÓN
# ============================================================

def evaluar(experiment, real_train, real_test):

    name = experiment["name"]

    output_dir = os.path.join(OUTPUT_ROOT, name)

    os.makedirs(output_dir, exist_ok=True)

    print()
    print("=" * 70)
    print(f"CONFIGURACIÓN: {name}")
    print("=" * 70)

    sinteticas = obtener_sinteticas(
        experiment,
        os.path.join(output_dir, "generated")
    )

    # --------------------------------------------------------
    # Nearest SSIM contra TEST (fidelidad)
    # --------------------------------------------------------

    print("Nearest SSIM sintético -> real TEST")

    v_test, idx_test = nearest_ssim_cruzado(
        sinteticas, real_test, "test"
    )

    # --------------------------------------------------------
    # Nearest SSIM contra TRAIN (memorización)
    # --------------------------------------------------------

    print("Nearest SSIM sintético -> real TRAIN")

    v_train, idx_train = nearest_ssim_cruzado(
        sinteticas, real_train, "train"
    )

    # --------------------------------------------------------
    # FID contra TEST
    # --------------------------------------------------------

    fid = calcular_fid(real_test, sinteticas)

    r_test = resumen(v_test)
    r_train = resumen(v_train)

    fila = {
        "config": name,
        "timesteps": experiment["timesteps"],
        "learning_rate": experiment["learning_rate"],
        "fid_test": fid,

        "nssim_test_mean": r_test["mean"],
        "nssim_test_std": r_test["std"],
        "nssim_test_median": r_test["median"],
        "nssim_test_max": r_test["max"],

        "nssim_train_mean": r_train["mean"],
        "nssim_train_std": r_train["std"],
        "nssim_train_max": r_train["max"],
        "pct_train_sobre_umbral": r_train["pct_sobre_umbral"],

        "gap_train_test": r_train["mean"] - r_test["mean"],
    }

    # --------------------------------------------------------
    # Guardar valores individuales
    # --------------------------------------------------------

    with open(
        os.path.join(output_dir, "nssim_values.csv"),
        "w", newline="", encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "index",
            "nssim_test",
            "idx_test_mas_cercano",
            "nssim_train",
            "idx_train_mas_cercano",
        ])

        for i in range(len(v_test)):
            writer.writerow([
                i,
                v_test[i], idx_test[i],
                v_train[i], idx_train[i],
            ])

    with open(
        os.path.join(output_dir, "nssim_summary.json"),
        "w", encoding="utf-8"
    ) as f:
        json.dump(fila, f, indent=4)

    print()
    print(f"FID (test)          : {fid:.4f}")
    print(
        f"nSSIM test  (media) : "
        f"{r_test['mean']:.4f} ± {r_test['std']:.4f}"
    )
    print(
        f"nSSIM train (media) : "
        f"{r_train['mean']:.4f} ± {r_train['std']:.4f}"
    )
    print(
        f"Muestras con nSSIM_train > {DUPLICADO_UMBRAL}: "
        f"{r_train['pct_sobre_umbral']:.1f}%"
    )

    return fila


# ============================================================
# TABLA LATEX
# ============================================================

def escribir_tabla_latex(filas, baseline, ruta):

    mejor = max(filas, key=lambda x: x["nssim_test_mean"])

    lineas = [
        r"\begin{table}[ht]",
        r"\caption{Nearest SSIM cruzado de las muestras sintéticas "
        r"frente al conjunto real, por configuración. "
        r"Media $\pm$ desviación estándar sobre 200 muestras "
        r"generadas. La columna TRAIN sirve como control de "
        r"memorización.}",
        r"\label{tab:nssim}",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Configuración & $T$ & LR & FID $\downarrow$ & "
        r"nSSIM test $\uparrow$ & nSSIM train \\",
        r"\midrule",
    ]

    for fila in filas:

        nombre = fila["config"].replace("_", r"\_")

        valor_test = (
            f"{fila['nssim_test_mean']:.4f} $\\pm$ "
            f"{fila['nssim_test_std']:.4f}"
        )

        if fila["config"] == mejor["config"]:
            valor_test = r"\textbf{" + valor_test + "}"

        lineas.append(
            f"{nombre} & {fila['timesteps']} & "
            f"{fila['learning_rate']:.1e} & "
            f"{fila['fid_test']:.2f} & "
            f"{valor_test} & "
            f"{fila['nssim_train_mean']:.4f} \\\\"
        )

    lineas += [
        r"\midrule",
        f"Referencia real (test vs.\\ train) & -- & -- & -- & "
        f"{baseline['mean']:.4f} $\\pm$ {baseline['std']:.4f} & -- \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    with open(ruta, "w", encoding="utf-8") as f:
        f.write("\n".join(lineas) + "\n")


# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    real_train, real_test = cargar_conjuntos()

    print(f"Real train: {tuple(real_train.shape)}")
    print(f"Real test : {tuple(real_test.shape)}")

    # --------------------------------------------------------
    # LÍNEA BASE: cuánto se parecen entre sí dos pulsos reales
    # --------------------------------------------------------

    print()
    print("Línea base: real TEST -> real TRAIN")

    v_base, _ = nearest_ssim_cruzado(
        real_test, real_train, "baseline"
    )

    baseline = resumen(v_base)

    print(
        f"Baseline nSSIM: "
        f"{baseline['mean']:.4f} ± {baseline['std']:.4f}"
    )

    with open(
        os.path.join(OUTPUT_ROOT, "baseline.json"),
        "w", encoding="utf-8"
    ) as f:
        json.dump(baseline, f, indent=4)

    # --------------------------------------------------------
    # GRILLA
    # --------------------------------------------------------

    filas = []

    for experiment in EXPERIMENTS:

        model_dir = os.path.join(
            EXPERIMENT_ROOT,
            experiment["name"],
            "best_model"
        )

        if not os.path.isdir(model_dir):
            print(
                f"[SALTADA] {experiment['name']}: "
                f"no hay best_model entrenado."
            )
            continue

        filas.append(
            evaluar(experiment, real_train, real_test)
        )

    if not filas:
        print("No se evaluó ninguna configuración.")
        return

    # --------------------------------------------------------
    # RESUMEN
    # --------------------------------------------------------

    summary_csv = os.path.join(OUTPUT_ROOT, "nssim_summary.csv")

    with open(
        summary_csv, "w", newline="", encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(f, fieldnames=filas[0].keys())

        writer.writeheader()
        writer.writerows(filas)

    escribir_tabla_latex(
        filas,
        baseline,
        os.path.join(OUTPUT_ROOT, "nssim_tabla.tex")
    )

    mejor = max(filas, key=lambda x: x["nssim_test_mean"])

    print()
    print("=" * 70)
    print("RESUMEN — nearest SSIM medio (sintético -> real test)")
    print("=" * 70)

    orden = sorted(
        filas,
        key=lambda x: x["nssim_test_mean"],
        reverse=True
    )

    for fila in orden:
        print(
            f"{fila['config']:<16} "
            f"nSSIM_test={fila['nssim_test_mean']:.4f}  "
            f"nSSIM_train={fila['nssim_train_mean']:.4f}  "
            f"gap={fila['gap_train_test']:+.4f}  "
            f"FID={fila['fid_test']:.2f}"
        )

    print()
    print(f"Referencia real-real: {baseline['mean']:.4f}")
    print(f"Mejor configuración : {mejor['config']}")

    if mejor["gap_train_test"] > 0.05:
        print(
            "AVISO: el nSSIM contra train supera claramente al de "
            "test en la mejor configuración; revise memorización "
            "antes de reportarla como la mejor."
        )

    print()
    print(f"CSV   : {summary_csv}")
    print(f"LaTeX : {os.path.join(OUTPUT_ROOT, 'nssim_tabla.tex')}")


if __name__ == "__main__":
    main()

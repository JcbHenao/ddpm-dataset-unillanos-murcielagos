import os
import csv
import gc
import random
import shutil
import time

from pathlib import Path
from dataclasses import dataclass

import numpy as np

import torch
import torch.nn.functional as F

from PIL import Image

from torch.utils.data import (
    Dataset,
    DataLoader,
    random_split
)

from diffusers import (
    UNet2DModel,
    DDPMScheduler,
    DDPMPipeline
)

from diffusers.optimization import (
    get_cosine_schedule_with_warmup
)

from accelerate import Accelerator
from accelerate.utils import (
    ProjectConfiguration,
    set_seed
)

from tqdm.auto import tqdm


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

@dataclass
class TrainingConfig:

    image_size = 128

    train_batch_size = 40
    eval_batch_size = 16

    num_epochs = 100

    gradient_accumulation_steps = 1

    warmup_steps = 80

    mixed_precision = "fp16"

    seed = 42

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset_dir = (
        "/media/adalab/bodega/CamiloBuitrago/"
        "espectrogramas/"
        "Muestra_Molossus_Alta_Calidad_128"
    )

    # --------------------------------------------------------
    # Salida general
    # --------------------------------------------------------

    output_root = (
        "ddpm-grid-experimentos"
    )


config = TrainingConfig()


# ============================================================
# MATRIZ EXPERIMENTAL
# ============================================================


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
# DATASET
# ============================================================

class EspectrogramasDataset(Dataset):

    def __init__(
        self,
        carpeta,
        transform=None
    ):

        self.carpeta = carpeta
        self.transform = transform

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


    def __len__(self):

        return len(self.archivos)


    def __getitem__(self, idx):

        ruta = os.path.join(

            self.carpeta,

            self.archivos[idx]

        )


        imagen = Image.open(
            ruta
        ).convert("RGB")


        if self.transform:

            imagen = self.transform(
                imagen
            )


        return {
            "images": imagen
        }


# ============================================================
# PREPROCESAMIENTO
# ============================================================

preprocess = torch.nn.Sequential()


from torchvision import transforms

preprocess = transforms.Compose([

    transforms.Resize(
        (
            config.image_size,
            config.image_size
        )
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5]
    ),

])


# ============================================================
# CREAR DATASET
# ============================================================

dataset = EspectrogramasDataset(

    config.dataset_dir,

    transform=preprocess

)


print()
print("=" * 80)
print("DATASET")
print("=" * 80)

print(
    f"Total de imágenes: {len(dataset)}"
)


# ============================================================
# VALIDAR DATASET
# ============================================================

if len(dataset) != 1000:

    raise ValueError(

        "El experimento requiere exactamente "
        "1000 imágenes."

    )


# ============================================================
# CREAR UN ÚNICO SPLIT
# ============================================================
#
# Este split se crea UNA SOLA VEZ y luego se utiliza en
# los cuatro experimentos.
#
# Esto garantiza que la comparación entre hiperparámetros
# se realice sobre exactamente las mismas imágenes.
#
# ============================================================

split_generator = torch.Generator().manual_seed(
    config.seed
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
print("DIVISIÓN DEL DATASET")
print("=" * 80)

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
# INFORMACIÓN DEL EXPERIMENTO
# ============================================================

print()
print("=" * 80)
print("MATRIZ EXPERIMENTAL")
print("=" * 80)

for experiment in EXPERIMENTS:

    print(

        f"{experiment['name']} | "
        f"timesteps={experiment['timesteps']} | "
        f"lr={experiment['learning_rate']}"

    )


# ============================================================
# CREAR MODELO
# ============================================================

def crear_modelo():

    model = UNet2DModel(

        sample_size=config.image_size,

        in_channels=3,

        out_channels=3,

        layers_per_block=2,

        block_out_channels=(

            128,
            128,
            256,
            256,
            512,
            512

        ),

        down_block_types=(

            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D"

        ),

        up_block_types=(

            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D"

        )

    )

    return model


# ============================================================
# CREAR DATALOADERS
# ============================================================

def crear_dataloaders():

    train_dataloader = DataLoader(

        train_dataset,

        batch_size=config.train_batch_size,

        shuffle=True,

        num_workers=4,

        pin_memory=True,

        persistent_workers=True

    )


    val_dataloader = DataLoader(

        val_dataset,

        batch_size=config.eval_batch_size,

        shuffle=False,

        num_workers=4,

        pin_memory=True,

        persistent_workers=True

    )


    test_dataloader = DataLoader(

        test_dataset,

        batch_size=config.eval_batch_size,

        shuffle=False,

        num_workers=4,

        pin_memory=True,

        persistent_workers=True

    )


    return (

        train_dataloader,
        val_dataloader,
        test_dataloader

    )


# ============================================================
# GUARDAR CSV
# ============================================================

def guardar_metricas(

    metricas,
    ruta_csv

):

    existe = os.path.exists(
        ruta_csv
    )


    with open(

        ruta_csv,

        "a",

        newline="",

        encoding="utf-8"

    ) as archivo:

        writer = csv.DictWriter(

            archivo,

            fieldnames=metricas.keys()

        )


        if not existe:

            writer.writeheader()


        writer.writerow(
            metricas
        )


# ============================================================
# LOSS DISTRIBUIDO
# ============================================================

@torch.no_grad()
def calcular_loss_dataset(

    model,

    dataloader,

    noise_scheduler,

    accelerator

):

    model.eval()


    total_loss = torch.tensor(

        0.0,

        device=accelerator.device

    )


    total_samples = torch.tensor(

        0,

        dtype=torch.long,

        device=accelerator.device

    )


    for batch in dataloader:

        clean_images = (
            batch["images"]
        )


        batch_size = (
            clean_images.shape[0]
        )


        noise = torch.randn_like(
            clean_images
        )


        timesteps = torch.randint(

            0,

            noise_scheduler.config.num_train_timesteps,

            (
                batch_size,
            ),

            device=clean_images.device

        ).long()


        noisy_images = (
            noise_scheduler.add_noise(

                clean_images,

                noise,

                timesteps

            )
        )


        noise_pred = model(

            noisy_images,

            timesteps,

            return_dict=False

        )[0]


        loss_per_sample = F.mse_loss(

            noise_pred,

            noise,

            reduction="none"

        ).mean(
            dim=(1, 2, 3)
        )


        total_loss += (
            loss_per_sample.sum()
        )


        total_samples += (
            batch_size
        )


    # --------------------------------------------------------
    # REDUCCIÓN ENTRE GPU
    # --------------------------------------------------------

    total_loss = accelerator.reduce(

        total_loss,

        reduction="sum"

    )


    total_samples = accelerator.reduce(

        total_samples,

        reduction="sum"

    )


    mean_loss = (

        total_loss
        /
        total_samples

    )


    model.train()


    return mean_loss.item()


# ============================================================
# TEST
# ============================================================

@torch.no_grad()
def calcular_test_loss(

    model,

    dataloader,

    noise_scheduler,

    accelerator

):

    return calcular_loss_dataset(

        model,

        dataloader,

        noise_scheduler,

        accelerator

    )


# ============================================================
# CREAR DIRECTORIOS
# ============================================================

def preparar_directorios(

    output_dir

):

    os.makedirs(
        output_dir,
        exist_ok=True
    )


    os.makedirs(

        os.path.join(
            output_dir,
            "metrics"
        ),

        exist_ok=True

    )


    os.makedirs(

        os.path.join(
            output_dir,
            "best_model"
        ),

        exist_ok=True

    )


    os.makedirs(

        os.path.join(
            output_dir,
            "samples"
        ),

        exist_ok=True

    )


# ============================================================
# ENTRENAR UN EXPERIMENTO
# ============================================================

def entrenar_experimento(

    accelerator,

    experiment

):

    name = experiment["name"]

    diffusion_timesteps = (
        experiment["timesteps"]
    )

    learning_rate = (
        experiment["learning_rate"]
    )


    output_dir = os.path.join(

        config.output_root,

        name

    )


    metrics_csv = os.path.join(

        output_dir,

        "metrics",

        "training_metrics.csv"

    )


    test_csv = os.path.join(

        output_dir,

        "metrics",

        "test_metrics.csv"

    )


    # ========================================================
    # SEMILLA
    # ========================================================

    set_seed(
        config.seed
    )


    # ========================================================
    # DIRECTORIOS
    # ========================================================

    if accelerator.is_main_process:

        preparar_directorios(
            output_dir
        )


    accelerator.wait_for_everyone()


    # ========================================================
    # DATALOADERS
    # ========================================================

    (

        train_dataloader,
        val_dataloader,
        test_dataloader

    ) = crear_dataloaders()


    # ========================================================
    # MODELO
    # ========================================================

    model = crear_modelo()


    # ========================================================
    # SCHEDULER DDPM
    # ========================================================

    noise_scheduler = DDPMScheduler(

        num_train_timesteps=
            diffusion_timesteps

    )


    # ========================================================
    # OPTIMIZADOR
    # ========================================================

    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=learning_rate

    )


    # ========================================================
    # LR SCHEDULER
    # ========================================================

    lr_scheduler = (
        get_cosine_schedule_with_warmup(

            optimizer=optimizer,

            num_warmup_steps=
                config.warmup_steps,

            num_training_steps=(

                len(train_dataloader)
                *
                config.num_epochs

            )

        )
    )


    # ========================================================
    # ACCELERATE
    # ========================================================

    (

        model,
        optimizer,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        lr_scheduler

    ) = accelerator.prepare(

        model,
        optimizer,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        lr_scheduler

    )


    # ========================================================
    # VARIABLES
    # ========================================================

    best_val_loss = float("inf")

    best_epoch = 0

    global_step = 0


    print()
    print("=" * 80)

    if accelerator.is_main_process:

        print(
            f"EXPERIMENTO: {name}"
        )

        print(
            f"Timesteps: {diffusion_timesteps}"
        )

        print(
            f"Learning rate: {learning_rate}"
        )

        print("=" * 80)


    # ========================================================
    # LOOP DE ENTRENAMIENTO
    # ========================================================

    for epoch in range(
        config.num_epochs
    ):

        epoch_start = time.time()

        model.train()


        # ----------------------------------------------------
        # ACUMULADORES
        # ----------------------------------------------------

        epoch_loss_sum = torch.tensor(

            0.0,

            device=accelerator.device

        )


        epoch_samples = torch.tensor(

            0,

            dtype=torch.long,

            device=accelerator.device

        )


        # ----------------------------------------------------
        # PROGRESS BAR
        # ----------------------------------------------------

        progress_bar = tqdm(

            total=len(train_dataloader),

            disable=(
                not accelerator.is_local_main_process
            )

        )


        progress_bar.set_description(

            f"{name} | "
            f"Epoch "
            f"{epoch + 1}/"
            f"{config.num_epochs}"

        )


        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        for batch in train_dataloader:

            clean_images = (
                batch["images"]
            )


            batch_size = (
                clean_images.shape[0]
            )


            noise = torch.randn_like(

                clean_images

            )


            timesteps = torch.randint(

                0,

                noise_scheduler.config.num_train_timesteps,

                (
                    batch_size,
                ),

                device=clean_images.device

            ).long()


            noisy_images = (
                noise_scheduler.add_noise(

                    clean_images,

                    noise,

                    timesteps

                )
            )


            with accelerator.accumulate(
                model
            ):

                noise_pred = model(

                    noisy_images,

                    timesteps,

                    return_dict=False

                )[0]


                loss = F.mse_loss(

                    noise_pred,

                    noise

                )


                accelerator.backward(
                    loss
                )


                accelerator.clip_grad_norm_(

                    model.parameters(),

                    1.0

                )


                optimizer.step()

                lr_scheduler.step()

                optimizer.zero_grad()


            # ------------------------------------------------
            # LOSS PARALELA
            # ------------------------------------------------

            epoch_loss_sum += (

                loss.detach()
                *
                batch_size

            )


            epoch_samples += (
                batch_size
            )


            current_lr = (
                lr_scheduler
                .get_last_lr()[0]
            )


            progress_bar.update(1)


            progress_bar.set_postfix(

                loss=f"{loss.item():.5f}",

                lr=f"{current_lr:.2e}"

            )


            global_step += 1


        progress_bar.close()


        # ----------------------------------------------------
        # REDUCCIÓN TRAIN LOSS
        # ----------------------------------------------------

        epoch_loss_sum = accelerator.reduce(

            epoch_loss_sum,

            reduction="sum"

        )


        epoch_samples = accelerator.reduce(

            epoch_samples,

            reduction="sum"

        )


        train_loss = (

            epoch_loss_sum
            /
            epoch_samples

        ).item()


        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        val_loss = calcular_loss_dataset(

            model,

            val_dataloader,

            noise_scheduler,

            accelerator

        )


        accelerator.wait_for_everyone()


        # ----------------------------------------------------
        # TIEMPO
        # ----------------------------------------------------

        epoch_time = (

            time.time()
            -
            epoch_start

        )


        current_lr = (
            lr_scheduler
            .get_last_lr()[0]
        )


        # ----------------------------------------------------
        # BEST MODEL
        # ----------------------------------------------------

        is_best = False


        if accelerator.is_main_process:

            if val_loss < best_val_loss:

                best_val_loss = val_loss

                best_epoch = epoch + 1

                is_best = True


                best_model_dir = os.path.join(

                    output_dir,

                    "best_model"

                )


                best_pipeline = DDPMPipeline(

                    unet=(

                        accelerator.unwrap_model(

                            model

                        )

                    ),

                    scheduler=noise_scheduler

                )


                best_pipeline.save_pretrained(

                    best_model_dir

                )


                print()
                print("-" * 80)

                print(
                    "NUEVO BEST MODEL"
                )

                print(
                    f"Experimento: "
                    f"{name}"
                )

                print(
                    f"Época: "
                    f"{best_epoch}"
                )

                print(
                    f"Val loss: "
                    f"{best_val_loss:.6f}"
                )

                print("-" * 80)


        # ----------------------------------------------------
        # BARRERA
        # ----------------------------------------------------

        accelerator.wait_for_everyone()


        # ----------------------------------------------------
        # MÉTRICAS
        # ----------------------------------------------------

        if accelerator.is_main_process:

            metricas = {

                "epoch":
                    epoch + 1,

                "train_loss":
                    train_loss,

                "val_loss":
                    val_loss,

                "learning_rate":
                    current_lr,

                "epoch_time_sec":
                    epoch_time,

                "global_step":
                    global_step,

                "is_best_model":
                    is_best,

                "best_epoch":
                    best_epoch,

                "best_val_loss":
                    best_val_loss,

                "timesteps":
                    diffusion_timesteps,

                "learning_rate_config":
                    learning_rate,

            }


            guardar_metricas(

                metricas,

                metrics_csv

            )


            print()

            print(
                f"[{name}] "
                f"Epoch {epoch + 1:02d} "
                f"| "
                f"Train {train_loss:.6f} "
                f"| "
                f"Val {val_loss:.6f} "
                f"| "
                f"LR {current_lr:.2e}"
            )


    # ========================================================
    # TEST DEL BEST MODEL
    # ========================================================

    accelerator.wait_for_everyone()


    if accelerator.is_main_process:

        best_model_dir = os.path.join(

            output_dir,

            "best_model"

        )


        print()
        print("=" * 80)
        print(
            f"EVALUACIÓN TEST: {name}"
        )
        print("=" * 80)


        # ----------------------------------------------------
        # CARGAR BEST PIPELINE
        # ----------------------------------------------------

        best_pipeline = (
            DDPMPipeline
            .from_pretrained(
                best_model_dir
            )
            .to(
                accelerator.device
            )
        )


        best_pipeline.unet.eval()

        test_loader_full = DataLoader(

            test_dataset,

            batch_size=config.eval_batch_size,

            shuffle=False,

            num_workers=4,

            pin_memory=True,

            persistent_workers=True

        )

        test_loss_total = 0.0

        test_total_samples = 0


        with torch.no_grad():

            for batch in test_loader_full:

                clean_images = (
                    batch["images"]
                    .to(
                        accelerator.device,
                        non_blocking=True
                    )
                )


                batch_size = (
                    clean_images.shape[0]
                )


                noise = torch.randn_like(
                    clean_images
                )


                timesteps = torch.randint(

                    0,

                    best_pipeline
                    .scheduler
                    .config
                    .num_train_timesteps,

                    (
                        batch_size,
                    ),

                    device=accelerator.device

                ).long()


                noisy_images = (
                    best_pipeline
                    .scheduler
                    .add_noise(

                        clean_images,

                        noise,

                        timesteps

                    )
                )


                noise_pred = (
                    best_pipeline
                    .unet(

                        noisy_images,

                        timesteps,

                        return_dict=False

                    )[0]
                )


                loss_per_sample = (
                    F.mse_loss(

                        noise_pred,

                        noise,

                        reduction="none"

                    )

                    .mean(
                        dim=(1, 2, 3)
                    )
                )


                test_loss_total += (
                    loss_per_sample
                    .sum()
                    .item()
                )


                test_total_samples += (
                    batch_size
                )


        test_loss = (
            test_loss_total
            /
            test_total_samples
        )


        # ----------------------------------------------------
        # GUARDAR TEST
        # ----------------------------------------------------

        guardar_metricas(

            {

                "test_loss":
                    test_loss,

                "num_test_images":
                    test_total_samples,

                "best_epoch":
                    best_epoch,

                "best_val_loss":
                    best_val_loss,

                "timesteps":
                    diffusion_timesteps,

                "learning_rate":
                    learning_rate,

            },

            test_csv

        )


        print()
        print(
            f"Best epoch     : "
            f"{best_epoch}"
        )

        print(
            f"Best val loss  : "
            f"{best_val_loss:.6f}"
        )

        print(
            f"Test loss      : "
            f"{test_loss:.6f}"
        )


    accelerator.wait_for_everyone()


    # ========================================================
    # LIMPIAR MEMORIA
    # ========================================================

    del model
    del optimizer
    del train_dataloader
    del val_dataloader
    del test_dataloader
    del lr_scheduler
    del noise_scheduler

    accelerator.free_memory()

    gc.collect()

    torch.cuda.empty_cache()


    accelerator.wait_for_everyone()


# ============================================================
# RESUMEN FINAL
# ============================================================

def construir_resumen():

    summary_path = os.path.join(

        config.output_root,

        "summary.csv"

    )


    filas = []


    for experiment in EXPERIMENTS:

        name = experiment["name"]


        test_csv = os.path.join(

            config.output_root,

            name,

            "metrics",

            "test_metrics.csv"

        )


        if not os.path.exists(
            test_csv
        ):

            continue


        with open(

            test_csv,

            "r",

            encoding="utf-8"

        ) as f:

            reader = csv.DictReader(f)

            row = next(reader)


        filas.append({

            "config":
                name,

            "timesteps":
                experiment["timesteps"],

            "learning_rate":
                experiment["learning_rate"],

            "best_epoch":
                row["best_epoch"],

            "best_val_loss":
                row["best_val_loss"],

            "test_loss":
                row["test_loss"],

            "best_model":
                os.path.join(

                    config.output_root,

                    name,

                    "best_model"

                )

        })


    # ========================================================
    # ESCRIBIR SUMMARY
    # ========================================================

    if len(filas) == 0:

        return


    with open(

        summary_path,

        "w",

        newline="",

        encoding="utf-8"

    ) as f:

        writer = csv.DictWriter(

            f,

            fieldnames=[

                "config",
                "timesteps",
                "learning_rate",
                "best_epoch",
                "best_val_loss",
                "test_loss",
                "best_model"

            ]

        )


        writer.writeheader()

        writer.writerows(
            filas
        )


    # ========================================================
    # MEJOR CONFIGURACIÓN
    # ========================================================

    best = min(

        filas,

        key=lambda x:
            float(x["best_val_loss"])

    )


    best_overall = os.path.join(

        config.output_root,

        "best_overall"

    )


    # --------------------------------------------------------
    # Crear enlace simbólico al mejor modelo
    # --------------------------------------------------------

    if os.path.lexists(
        best_overall
    ):

        os.unlink(
            best_overall
        )


    os.symlink(

        os.path.abspath(
            best["best_model"]
        ),

        best_overall

    )


    # ========================================================
    # REPORTE
    # ========================================================

    print()
    print("=" * 80)
    print("RESUMEN FINAL")
    print("=" * 80)


    for row in filas:

        print()

        print(
            f"{row['config']}"
        )

        print(
            f"  Timesteps    : "
            f"{row['timesteps']}"
        )

        print(
            f"  Learning rate: "
            f"{row['learning_rate']}"
        )

        print(
            f"  Best epoch   : "
            f"{row['best_epoch']}"
        )

        print(
            f"  Best val     : "
            f"{float(row['best_val_loss']):.6f}"
        )

        print(
            f"  Test loss    : "
            f"{float(row['test_loss']):.6f}"
        )


    print()
    print("=" * 80)

    print(
        "MEJOR CONFIGURACIÓN"
    )

    print(
        f"Config       : "
        f"{best['config']}"
    )

    print(
        f"Timesteps    : "
        f"{best['timesteps']}"
    )

    print(
        f"Learning rate: "
        f"{best['learning_rate']}"
    )

    print(
        f"Best epoch   : "
        f"{best['best_epoch']}"
    )

    print(
        f"Best val     : "
        f"{float(best['best_val_loss']):.6f}"
    )

    print(
        f"Test loss    : "
        f"{float(best['test_loss']):.6f}"
    )

    print()
    print(
        f"Summary: "
        f"{summary_path}"
    )

    print(
        f"Best overall: "
        f"{best_overall}"
    )

    print("=" * 80)


# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs(

        config.output_root,

        exist_ok=True

    )


    logging_dir = os.path.join(

        config.output_root,

        "logs"

    )


    accelerator_project_config = (
        ProjectConfiguration(

            project_dir=
                config.output_root,

            logging_dir=
                logging_dir

        )
    )


    # ========================================================
    # ACCELERATE
    # ========================================================

    accelerator = Accelerator(

        mixed_precision=
            config.mixed_precision,

        gradient_accumulation_steps=
            config.gradient_accumulation_steps,

        project_config=
            accelerator_project_config,

        log_with="tensorboard"

    )


    if accelerator.is_main_process:

        accelerator.init_trackers(
            "ddpm_grid_experimentos"
        )


    accelerator.wait_for_everyone()


    print()

    if accelerator.is_main_process:

        print("=" * 80)
        print("DDPM GRID EXPERIMENT")
        print("=" * 80)

        print(
            f"Procesos: "
            f"{accelerator.num_processes}"
        )

        print(
            f"Dispositivo: "
            f"{accelerator.device}"
        )
        print("=" * 80)


    # ========================================================
    # EJECUTAR LAS CONFIGURACIONES
    # ========================================================

    for experiment in EXPERIMENTS:

        entrenar_experimento(

            accelerator,

            experiment

        )


        accelerator.wait_for_everyone()


    # ========================================================
    # CONSTRUIR RESUMEN
    # ========================================================

    if accelerator.is_main_process:

        construir_resumen()


    accelerator.wait_for_everyone()


    if accelerator.is_main_process:

        accelerator.end_training()


# ============================================================
# EJECUCIÓN
# ============================================================

if __name__ == "__main__":

    main()
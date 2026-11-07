import os
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from PIL import Image

from torchvision import transforms

from torch.utils.data import Dataset, DataLoader

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
    ProjectConfiguration
)

from huggingface_hub import (
    create_repo,
    upload_folder
)

from tqdm.auto import tqdm


# ============================================================
# CONFIGURACIÓN
# ============================================================

@dataclass
class TrainingConfig:

    image_width = 2048
    image_height = 128

    train_batch_size = 2
    eval_batch_size = 2

    num_epochs = 100

    num_train_timesteps = 1500

    gradient_accumulation_steps = 1

    learning_rate = 2e-4

    # 10 % del entrenamiento
    warmup_ratio = 0.10

    save_image_epochs = 10
    save_model_epochs = 30

    mixed_precision = "bf16"

    output_dir = (
        "ddpm-pulsos-saccopteryx-2048-1500ntt"
    )

    push_to_hub = False
    hub_private_repo = False
    overwrite_output_dir = True

    seed = 42


config = TrainingConfig()


# ============================================================
# DATASET
# ============================================================

RUTA_DATASET_ROOT = (
    "/media/adalab/bodega/CamiloBuitrago/"
    "espectrogramas/"
    "Espectrogramas_Secuencias_Chunks_1.5s_2048x128px/"
    "Dataset_DDPM_2048x128px/"
)

RUTA_DATASET_LOCAL = os.path.join(
    RUTA_DATASET_ROOT,
    "train"
)

RUTA_DATASET_VAL = os.path.join(
    RUTA_DATASET_ROOT,
    "validation"
)


class EspectrogramasDataset(Dataset):

    def __init__(
        self,
        carpeta,
        transform=None
    ):

        self.carpeta = Path(carpeta)
        self.transform = transform

        if not self.carpeta.exists():

            raise ValueError(
                "\nLa carpeta no existe:\n"
                f"{self.carpeta}\n"
            )

        if not self.carpeta.is_dir():

            raise ValueError(
                "\nLa ruta no es un directorio:\n"
                f"{self.carpeta}\n"
            )

        self.archivos = sorted(
            [
                f
                for f in self.carpeta.rglob("*")
                if (
                    f.is_file()
                    and
                    f.suffix.lower() == ".png"
                )
            ]
        )

        if len(self.archivos) == 0:

            raise ValueError(
                "\nNo se encontraron imagenes .png en:\n"
                f"{self.carpeta}\n\n"
                "Comprueba la estructura del dataset."
            )

    def __len__(self):

        return len(self.archivos)

    def __getitem__(self, idx):

        ruta = self.archivos[idx]

        imagen = Image.open(
            ruta
        ).convert("RGB")

        if imagen.size != (
            config.image_width,
            config.image_height
        ):

            raise ValueError(
                "\nDimensiones incorrectas:\n"
                f"{ruta}\n"
                f"Esperado: "
                f"{config.image_width}x"
                f"{config.image_height}\n"
                f"Encontrado: "
                f"{imagen.size}"
            )

        if self.transform:

            imagen = self.transform(
                imagen
            )

        return {
            "images": imagen
        }


# ============================================================
# TRANSFORMACIONES
# ============================================================

preprocess = transforms.Compose(
    [
        transforms.ToTensor(),

        transforms.Normalize(
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5]
        ),
    ]
)


# ============================================================
# CREACIÓN DE DATASETS
# ============================================================

dataset = EspectrogramasDataset(
    RUTA_DATASET_LOCAL,
    transform=preprocess
)

val_dataset = EspectrogramasDataset(
    RUTA_DATASET_VAL,
    transform=preprocess
)


print(
    f"Imagenes cargadas para entrenamiento: "
    f"{len(dataset)}"
)

print(
    f"Imagenes cargadas para validacion: "
    f"{len(val_dataset)}"
)


# ============================================================
# DATALOADERS
# ============================================================

train_dataloader = DataLoader(
    dataset,
    batch_size=config.train_batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)


val_dataloader = DataLoader(
    val_dataset,
    batch_size=config.eval_batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)
# ============================================================
# VALIDATION LOSS
# ============================================================

@torch.no_grad()
def calculate_validation_loss(
    accelerator,
    model,
    noise_scheduler,
    val_dataloader
):

    model.eval()

    total_loss = torch.tensor(
        0.0,
        device=accelerator.device
    )

    total_elements = torch.tensor(
        0.0,
        device=accelerator.device
    )

    for batch in val_dataloader:

        clean_images = batch["images"]

        noise = torch.randn_like(
            clean_images
        )

        batch_size = clean_images.shape[0]

        timesteps = torch.randint(
            0,
            noise_scheduler.num_train_timesteps,
            (batch_size,),
            device=clean_images.device
        ).long()

        noisy_images = noise_scheduler.add_noise(
            clean_images,
            noise,
            timesteps
        )

        noise_pred = model(
            noisy_images,
            timesteps,
            return_dict=False
        )[0]

        loss = F.mse_loss(
            noise_pred,
            noise,
            reduction="sum"
        )

        total_loss += loss

        total_elements += noise.numel()

    total_loss = accelerator.reduce(
        total_loss,
        reduction="sum"
    )

    total_elements = accelerator.reduce(
        total_elements,
        reduction="sum"
    )

    model.train()

    return (
        total_loss / total_elements
    ).item()
    
# =====================================================
# Modelo UNet2D                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
# =====================================================
# https://arxiv.org/pdf/2006.11239
 
model = UNet2DModel(
    sample_size=(
    config.image_height,
    config.image_width
),
    in_channels=3,   # espectrogramas con cmap='jet' -> RGB de 3 canales
    out_channels=3,
    layers_per_block=2,
    block_out_channels=(128, 128, 256, 256, 512, 512),
    down_block_types=(
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",
        "AttnUpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),
)
 

 
sample_image = dataset[0]["images"].unsqueeze(0)
print("Input shape:", sample_image.shape)
 
 
# =====================================================
# Scheduler de ruido
# =====================================================
 
noise_scheduler = DDPMScheduler(
    num_train_timesteps=config.num_train_timesteps
)


# ============================================================
# OPTIMIZADOR
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.learning_rate
)

num_update_steps_per_epoch = math.ceil(
    len(train_dataloader)
    / config.gradient_accumulation_steps
)

num_training_steps = (
    config.num_epochs
    * num_update_steps_per_epoch
)

num_warmup_steps = max(
    1,
    int(
        num_training_steps
        * config.warmup_ratio
    )
)

lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps,
)

 
# =====================================================
# Utilidades de evaluacion
# =====================================================
def make_grid(
    images,
    rows,
    cols
):

    w, h = images[0].size

    grid = Image.new(
        "RGB",
        size=(
            cols * w,
            rows * h
        )
    )

    for i, image in enumerate(images):

        grid.paste(
            image,
            box=(
                i % cols * w,
                i // cols * h
            )
        )

    return grid


def build_pipeline(
    accelerator,
    model,
    noise_scheduler
):

    unwrapped_model = accelerator.unwrap_model(
        model
    )

    pipeline = DDPMPipeline(
        unet=unwrapped_model,
        scheduler=noise_scheduler
    )

    pipeline = pipeline.to(
        accelerator.device
    )

    return pipeline


def evaluate(
    config,
    epoch_name,
    pipeline,
    accelerator
):

    generator = torch.Generator(
        device=accelerator.device
    ).manual_seed(
        config.seed
    )

    pipeline.unet.eval()

    with torch.inference_mode():

        images = pipeline(
            batch_size=config.eval_batch_size,
            generator=generator,
            output_type="pil"
        ).images

    cols = min(
        4,
        len(images)
    )

    rows = math.ceil(
        len(images) / cols
    )

    image_grid = make_grid(
        images,
        rows,
        cols
    )

    test_dir = os.path.join(
        config.output_dir,
        "samples"
    )

    os.makedirs(
        test_dir,
        exist_ok=True
    )

    output_path = os.path.join(
        test_dir,
        f"{epoch_name}.png"
    )

    image_grid.save(
        output_path
    )

    print(
        f"Muestras guardadas en: "
        f"{output_path}"
    )

# =====================================================
# Loop de entrenamiento 
# =====================================================
# ============================================================
# LOOP DE ENTRENAMIENTO
# ============================================================

def train_loop(
    config,
    model,
    noise_scheduler,
    optimizer,
    train_dataloader,
    val_dataloader
):

    best_val_loss = float("inf")

    best_epoch = -1


    # ========================================================
    # ACCELERATOR
    # ========================================================

    logging_dir = os.path.join(
        config.output_dir,
        "logs"
    )

    accelerator_project_config = ProjectConfiguration(
        project_dir=config.output_dir,
        logging_dir=logging_dir
    )

    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,

        gradient_accumulation_steps=(
            config.gradient_accumulation_steps
        ),

        log_with="tensorboard",

        project_config=(
            accelerator_project_config
        ),
    )


    # ========================================================
    # DIRECTORIO DE SALIDA
    # ========================================================

    if accelerator.is_main_process:

        if config.push_to_hub:

            repo_id = create_repo(
                repo_id=Path(
                    config.output_dir
                ).name,

                exist_ok=True
            ).repo_id

        elif config.output_dir is not None:

            os.makedirs(
                config.output_dir,
                exist_ok=True
            )

        accelerator.init_trackers(
            "train"
        )


    # ========================================================
    # PREPARAR MODELO / OPTIMIZADOR / DATALOADERS
    # ========================================================

    (
        model,
        optimizer,
        train_dataloader,
        val_dataloader
    ) = accelerator.prepare(
        model,
        optimizer,
        train_dataloader,
        val_dataloader
    )


    # ========================================================
    # CALCULAR STEPS REALES
    # ========================================================

    num_update_steps_per_epoch = math.ceil(

        len(train_dataloader)

        /
        
        config.gradient_accumulation_steps
    )

    num_training_steps = (

        config.num_epochs
        *
        num_update_steps_per_epoch
    )


    # ========================================================
    # WARM-UP = 10 %
    # ========================================================

    num_warmup_steps = max(

        1,

        int(
            num_training_steps
            *
            config.warmup_ratio
        )
    )


    # ========================================================
    # LEARNING RATE SCHEDULER
    # ========================================================

    lr_scheduler = get_cosine_schedule_with_warmup(

        optimizer=optimizer,

        num_warmup_steps=num_warmup_steps,

        num_training_steps=num_training_steps,
    )


    lr_scheduler = accelerator.prepare(
        lr_scheduler
    )


    # ========================================================
    # CONFIGURACIÓN
    # ========================================================

    if accelerator.is_main_process:

        print()
        print("=" * 70)
        print("CONFIGURACIÓN DEL ENTRENAMIENTO")
        print("=" * 70)

        print(
            f"Imagenes entrenamiento : "
            f"{len(dataset)}"
        )

        print(
            f"Imagenes validacion    : "
            f"{len(val_dataset)}"
        )

        print(
            f"Resolución             : "
            f"{config.image_width}x"
            f"{config.image_height}"
        )

        print(
            f"Batch por proceso      : "
            f"{config.train_batch_size}"
        )

        print(
            f"Épocas                 : "
            f"{config.num_epochs}"
        )

        print(
            f"Timesteps              : "
            f"{config.num_train_timesteps}"
        )

        print(
            f"Learning rate          : "
            f"{config.learning_rate}"
        )

        print(
            f"Warm-up                : "
            f"{config.warmup_ratio * 100:.1f}%"
        )

        print(
            f"Steps por época        : "
            f"{num_update_steps_per_epoch}"
        )

        print(
            f"Steps totales         : "
            f"{num_training_steps}"
        )

        print(
            f"Warm-up steps          : "
            f"{num_warmup_steps}"
        )

        print("=" * 70)
        print()


    global_step = 0


    # ========================================================
    # EPOCHS
    # ========================================================

    for epoch in range(
        config.num_epochs
    ):

        model.train()


        progress_bar = tqdm(

            total=len(train_dataloader),

            disable=(
                not accelerator.is_local_main_process
            ),
        )

        progress_bar.set_description(
            f"Epoch {epoch + 1}/{config.num_epochs}"
        )


        # ====================================================
        # TRAIN
        # ====================================================

        for step, batch in enumerate(
            train_dataloader
        ):

            clean_images = batch["images"]


            noise = torch.randn_like(
                clean_images
            )


            batch_size = (
                clean_images.shape[0]
            )


            timesteps = torch.randint(

                0,

                noise_scheduler.num_train_timesteps,

                (
                    batch_size,
                ),

                device=clean_images.device,
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

                optimizer.zero_grad(
                    set_to_none=True
                )


            progress_bar.update(1)


            logs = {

                "train_loss":
                    loss.detach().item(),

                "lr":
                    lr_scheduler.get_last_lr()[0],

                "step":
                    global_step,
            }


            progress_bar.set_postfix(
                **logs
            )


            accelerator.log(
                logs,
                step=global_step
            )


            global_step += 1


        progress_bar.close()


        # ====================================================
        # VALIDATION
        # ====================================================

        val_loss = calculate_validation_loss(

            accelerator,

            model,

            noise_scheduler,

            val_dataloader
        )


        accelerator.log(

            {
                "val_loss": val_loss
            },

            step=global_step
        )


        # ====================================================
        # RESULTADOS DE LA ÉPOCA
        # ====================================================

        if accelerator.is_main_process:

            print(
                f"\nEpoch {epoch + 1}"
            )

            print(
                f"Train Loss : "
                f"{loss.detach().item():.6f}"
            )

            print(
                f"Val Loss   : "
                f"{val_loss:.6f}"
            )


        # ====================================================
        # MEJOR MODELO
        # ====================================================

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            best_epoch = epoch + 1


            if accelerator.is_main_process:

                best_dir = os.path.join(

                    config.output_dir,

                    "best_model"
                )

                pipeline = build_pipeline(

                    accelerator,

                    model,

                    noise_scheduler
                )

                pipeline.save_pretrained(
                    best_dir
                )

                print(
                    f"Nuevo mejor modelo:"
                    f" epoch={best_epoch},"
                    f" val_loss={best_val_loss:.6f}"
                )


        # ====================================================
        # GENERACIÓN DE MUESTRAS
        # ====================================================

        if accelerator.is_main_process:

            pipeline = build_pipeline(

                accelerator,

                model,

                noise_scheduler
            )


            if (

                (epoch + 1)
                % config.save_image_epochs
                == 0

                or

                epoch
                ==
                config.num_epochs - 1

            ):

                evaluate(

                    config,

                    f"epoch_{epoch + 1:04d}",

                    pipeline,

                    accelerator
                )


            # =================================================
            # GUARDADO DEL MODELO
            # =================================================

            if (

                (epoch + 1)
                % config.save_model_epochs
                == 0

                or

                epoch
                ==
                config.num_epochs - 1

            ):

                if config.push_to_hub:

                    upload_folder(

                        repo_id=repo_id,

                        folder_path=(
                            config.output_dir
                        ),

                        commit_message=(
                            f"Epoch {epoch + 1}"
                        ),

                        ignore_patterns=[
                            "step_*",
                            "epoch_*"
                        ],
                    )

                else:

                    pipeline.save_pretrained(

                        config.output_dir
                    )


    # ========================================================
    # RESULTADO FINAL
    # ========================================================

    if accelerator.is_main_process:

        print()
        print("=" * 70)
        print("ENTRENAMIENTO FINALIZADO")
        print("=" * 70)

        print(
            f"Best epoch    : "
            f"{best_epoch}"
        )

        print(
            f"Best val loss : "
            f"{best_val_loss:.6f}"
        )

        print("=" * 70)
        print()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    train_loop(

        config,

        model,

        noise_scheduler,

        optimizer,

        train_dataloader,

        val_dataloader
    )
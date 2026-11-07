import os
import pandas as pd

from dataclasses import dataclass
from pathlib import Path
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from diffusers import (
    DiTTransformer2DModel,
    DDPMScheduler,
)

from diffusers.optimization import (
    get_cosine_schedule_with_warmup
)

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration

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

    image_size = 128

    train_batch_size = 20
    eval_batch_size = 16

    num_epochs = 300

    gradient_accumulation_steps = 1

    learning_rate = 1e-4

    lr_warmup_steps = 200

    save_image_epochs = 10
    save_model_epochs = 50

    mixed_precision = "fp16"

    output_dir = (
        "dit-pulsos-molossus-128-500ntt"
    )

    seed = 42

    num_train_timesteps = 500
    push_to_hub = False
    hub_private_repo = False
    overwrite_output_dir = True

config = TrainingConfig()


# ============================================================
# RUTA DATASET
# ============================================================

RUTA_DATASET_LOCAL = (
    "/media/adalab/bodega/CamiloBuitrago/espectrogramas/Muestra_Molossus_Alta_Calidad_128"
)

RUTA_CSV = (
    "/media/adalab/bodega/CamiloBuitrago/"
    "espectrogramas/Muestra_Molossus_Alta_Calidad_128/"
    "_muestra_seleccionada.csv"
)
# ============================================================
# DATASET
# ============================================================

class EspectrogramasDataset(Dataset):

    def __init__(
        self,
        carpeta,
        csv_path,
        transform=None
    ):

        self.carpeta = carpeta
        self.transform = transform

        # ====================================================
        # Leer CSV
        # ====================================================

        self.df = pd.read_csv(csv_path)

        # ====================================================
        # Verificar columnas necesarias
        # ====================================================

        columnas_requeridas = [
            "archivo_imagen",
            "especie"
        ]

        for columna in columnas_requeridas:

            if columna not in self.df.columns:

                raise ValueError(
                    f"No se encontró la columna "
                    f"'{columna}' en el CSV.\n"
                    f"Columnas disponibles: "
                    f"{self.df.columns.tolist()}"
                )

        # ====================================================
        # Conservar únicamente lo necesario
        # ====================================================

        self.df = self.df[
            [
                "archivo_imagen",
                "especie"
            ]
        ].copy()

        # ====================================================
        # Limpiar datos
        # ====================================================

        self.df["archivo_imagen"] = (
            self.df["archivo_imagen"]
            .astype(str)
            .str.strip()
        )

        self.df["especie"] = (
            self.df["especie"]
            .astype(str)
            .str.strip()
        )

        # Eliminar filas incompletas

        self.df = self.df[
            (self.df["archivo_imagen"] != "") &
            (self.df["especie"] != "")
        ].copy()

        # ====================================================
        # Crear mapping especie -> label
        # ====================================================

        self.especies = sorted(
            self.df["especie"].unique()
        )

        self.label_to_id = {
            especie: idx
            for idx, especie
            in enumerate(self.especies)
        }

        self.id_to_label = {
            idx: especie
            for especie, idx
            in self.label_to_id.items()
        }

        # ====================================================
        # Verificar que los PNG existan
        # ====================================================

        filas_validas = []

        for _, row in self.df.iterrows():

            nombre_archivo = row["archivo_imagen"]

            ruta = os.path.join(
                self.carpeta,
                nombre_archivo
            )

            if os.path.isfile(ruta):

                filas_validas.append(row)

        self.df = (
            pd.DataFrame(filas_validas)
            .reset_index(drop=True)
        )

        if len(self.df) == 0:

            raise ValueError(
                "Ninguna de las imágenes indicadas "
                "en el CSV existe en la carpeta."
            )

        # ====================================================
        # Información del dataset
        # ====================================================

        print("\n============================================")
        print("DATASET CONDICIONADO POR ESPECIE")
        print("============================================")

        print(
            f"Imágenes válidas : {len(self.df)}"
        )

        print(
            f"Número de clases : {len(self.especies)}"
        )

        print("\nClases:")

        for label, especie in self.id_to_label.items():

            cantidad = (
                self.df[
                    self.df["especie"] == especie
                ]
                .shape[0]
            )

            print(
                f"{label:2d} -> "
                f"{especie:<30} "
                f"({cantidad} imágenes)"
            )

        print("============================================\n")

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        nombre_archivo = row[
            "archivo_imagen"
        ]

        especie = row[
            "especie"
        ]

        # ----------------------------------------------------
        # Label numérico
        # ----------------------------------------------------

        class_label = self.label_to_id[
            especie
        ]

        # ----------------------------------------------------
        # Ruta imagen
        # ----------------------------------------------------

        ruta = os.path.join(
            self.carpeta,
            nombre_archivo
        )

        # ----------------------------------------------------
        # Cargar imagen
        # ----------------------------------------------------

        imagen = (
            Image.open(ruta)
            .convert("RGB")
        )

        if self.transform:

            imagen = self.transform(imagen)

        return {
            "images": imagen,

            "class_labels": torch.tensor(
                class_label,
                dtype=torch.long
            ),
        }

# ============================================================
# PREPROCESAMIENTO
# ============================================================

preprocess = transforms.Compose(
    [

        transforms.Resize(
            (
                config.image_size,
                config.image_size
            )
        ),

        transforms.ToTensor(),

        # RGB -> [-1, 1]
        transforms.Normalize(
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5]
        ),
    ]
)


# ============================================================
# DATASET
# ============================================================

dataset = EspectrogramasDataset(
    carpeta=RUTA_DATASET_LOCAL,
    csv_path=RUTA_CSV,
    transform=preprocess
)


print(
    f"Imagenes cargadas para entrenamiento: "
    f"{len(dataset)}"
)


# ============================================================
# DATALOADER
# ============================================================

train_dataloader = DataLoader(
    dataset,
    batch_size=config.train_batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)


# ============================================================
# DIFFUSION TRANSFORMER
# ============================================================
#
# 128 x 128
#
# patch_size = 8
#
# => 128 / 8 = 16
#
# => 16 x 16 = 256 tokens
#
# embedding dimension:
#
# 6 heads x 64 = 384
#
# ============================================================
NUM_CLASSES = len(dataset.especies)

print(
    f"Número de clases para el DiT: {NUM_CLASSES}"
)


model = DiTTransformer2DModel(
    sample_size=config.image_size,

    in_channels=3,
    out_channels=3,

    patch_size=2,

    num_layers=8,

    num_attention_heads=6,
    attention_head_dim=64,

    activation_fn="gelu-approximate",

    num_embeds_ada_norm=(
        config.num_train_timesteps
    ),

    norm_type="ada_norm_zero",

    norm_elementwise_affine=False,

    attention_bias=True,

    dropout=0.0,
)

# ============================================================
# INFORMACIÓN DEL MODELO
# ============================================================

num_parameters = sum(
    p.numel()
    for p in model.parameters()
)

num_trainable_parameters = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)


print("\n============================================")
print("DIFFUSION TRANSFORMER")
print("============================================")

print(
    f"Image size             : "
    f"{config.image_size}x{config.image_size}"
)

print(
    f"Channels               : 3"
)

print(
    f"Patch size             : 8"
)

print(
    f"Tokens                 : "
    f"{(config.image_size // 8) ** 2}"
)

print(
    f"Transformer layers     : "
    f"{model.config.num_layers}"
)

print(
    f"Attention heads        : "
    f"{model.config.num_attention_heads}"
)

print(
    f"Attention head dim     : "
    f"{model.config.attention_head_dim}"
)

print(
    f"Embedding dimension    : "
    f"{model.config.num_attention_heads * model.config.attention_head_dim}"
)

print(
    f"Total parameters       : "
    f"{num_parameters:,}"
)

print(
    f"Trainable parameters   : "
    f"{num_trainable_parameters:,}"
)

print("============================================\n")


# ============================================================
# VERIFICACIÓN DEL INPUT
# ============================================================

sample_image = (
    dataset[0]["images"]
    .unsqueeze(0)
)


print(
    "Input shape:",
    sample_image.shape
)


# ============================================================
# SCHEDULER DDPM
# ============================================================

noise_scheduler = DDPMScheduler(
    num_train_timesteps=(
        config.num_train_timesteps
    )
)


# ============================================================
# PRUEBA FORWARD
# ============================================================

noise = torch.randn_like(
    sample_image
)
timesteps = torch.tensor(
    [50],
    dtype=torch.long
)

class_labels = torch.tensor(
    [
        dataset.label_to_id[
            dataset.especies[0]
        ]
    ],
    dtype=torch.long
)

noisy_image = noise_scheduler.add_noise(
    sample_image,
    noise,
    timesteps
)

with torch.no_grad():

    noise_pred = model(
        noisy_image,
        timestep=timesteps,
        class_labels=class_labels
    ).sample

loss = F.mse_loss(
    noise_pred,
    noise
)

print(
    "Input noisy image:",
    noisy_image.shape
)

print(
    "Noise prediction:",
    noise_pred.shape
)

print(
    f"Forward loss      : "
    f"{loss.item():.6f}"
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.learning_rate,
    weight_decay=0.0
)

# ============================================================
# LR SCHEDULER
# ============================================================

lr_scheduler = get_cosine_schedule_with_warmup(

    optimizer=optimizer,

    num_warmup_steps=(
        config.lr_warmup_steps
    ),

    num_training_steps=(
        len(train_dataloader)
        * config.num_epochs
    ),
)


# ============================================================
# GENERACIÓN DE IMÁGENES
# ============================================================



@torch.no_grad()
def generate_images(
    model,
    scheduler,
    num_images,
    image_size,
    device,
    class_label
    ):



    model.eval()

    # --------------------------------------------------------
    # Ruido gaussiano inicial
    # --------------------------------------------------------

    images = torch.randn(
        (
            num_images,
            3,
            image_size,
            image_size
        ),
        device=device
    )

    # --------------------------------------------------------
    # Scheduler de inferencia
    # --------------------------------------------------------

    scheduler.set_timesteps(
        scheduler.config.num_train_timesteps,
        device=device
    )
    class_labels = torch.full(
        (num_images,),
        class_label,
        device=device,
        dtype=torch.long
    )
    # --------------------------------------------------------
    # Reverse diffusion
    # --------------------------------------------------------

    for timestep in scheduler.timesteps:

        timesteps = torch.full(
            (
                num_images,
            ),
            timestep,
            device=device,
            dtype=torch.long
        )

        noise_pred = model(
            images,
            timestep=timesteps,
            class_labels=class_labels
        ).sample

        step_output = scheduler.step(
            noise_pred,
            timestep,
            images
        )

        images = step_output.prev_sample

    # --------------------------------------------------------
    # [-1,1] -> [0,1]
    # --------------------------------------------------------

    images = (
        images.clamp(-1, 1)
        + 1
    ) / 2

    return images


# ============================================================
# MAKE GRID
# ============================================================

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
                (i % cols) * w,
                (i // cols) * h
            )
        )

    return grid


# ============================================================
# EVALUACIÓN
# ============================================================
def evaluate(
    config,
    epoch,
    model,
    scheduler,
    accelerator,
    dataset
):

    model.eval()

    test_dir = os.path.join(
        config.output_dir,
        "samples"
    )

    os.makedirs(
        test_dir,
        exist_ok=True
    )

    for especie, class_label in dataset.label_to_id.items():

        images_tensor = generate_images(
            model=model,
            scheduler=scheduler,
            num_images=config.eval_batch_size,
            image_size=config.image_size,
            device=accelerator.device,
            class_label=class_label
        )

        images = []

        for image_tensor in images_tensor:

            image_tensor = (
                image_tensor
                .detach()
                .cpu()
            )

            image_tensor = (
                image_tensor
                .permute(1, 2, 0)
                .numpy()
            )

            image = Image.fromarray(
                (
                    image_tensor * 255
                )
                .clip(0, 255)
                .astype("uint8")
            )

            images.append(image)

        image_grid = make_grid(
            images,
            rows=4,
            cols=4
        )

        nombre_seguro = (
            especie
            .replace("/", "_")
            .replace(" ", "_")
        )

        ruta_salida = os.path.join(
            test_dir,
            f"{epoch:04d}_{nombre_seguro}.png"
        )

        image_grid.save(
            ruta_salida
        )

        print(
            f"Muestras generadas para "
            f"{especie}: {ruta_salida}"
        )

    model.train()

# ============================================================
# TRAIN LOOP
# ============================================================

def train_loop(
    config,
    model,
    noise_scheduler,
    optimizer,
    train_dataloader,
    lr_scheduler,
    dataset
):

    logging_dir = os.path.join(
        config.output_dir,
        "logs"
    )

    accelerator_project_config = (
        ProjectConfiguration(
            project_dir=config.output_dir,
            logging_dir=logging_dir
        )
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
    # MAIN PROCESS
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
            "train_dit_pulsos_molossus"
        )

    # ========================================================
    # PREPARAR CON ACCELERATE
    # ========================================================

    (
        model,
        optimizer,
        train_dataloader,
        lr_scheduler
    ) = accelerator.prepare(
        model,
        optimizer,
        train_dataloader,
        lr_scheduler
    )

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
            )
        )

        progress_bar.set_description(
            f"Epoch {epoch + 1}"
        )

        # ====================================================
        # BATCHES
        # ====================================================

        for step, batch in enumerate(
            train_dataloader
        ):

            clean_images = batch[
                "images"
            ]

            class_labels = batch[
                "class_labels"
            ].to(
                clean_images.device
            )

            # ------------------------------------------------
            # Ruido
            # ------------------------------------------------

            noise = torch.randn_like(
                clean_images
            )

            bs = clean_images.shape[0]

            # ------------------------------------------------
            # Timesteps aleatorios
            # ------------------------------------------------

            timesteps = torch.randint(

                0,

                noise_scheduler.num_train_timesteps,

                (bs,),

                device=clean_images.device
            ).long()

            # ------------------------------------------------
            # Forward diffusion
            # ------------------------------------------------

            noisy_images = (
                noise_scheduler.add_noise(
                    clean_images,
                    noise,
                    timesteps
                )
            )

            # ------------------------------------------------
            # Transformer
            # ------------------------------------------------

            with accelerator.accumulate(
                model
            ):

                noise_pred = model(
                    noisy_images,
                    timestep=timesteps,
                    class_labels=class_labels,
                    return_dict=False
                )[0]

                # --------------------------------------------
                # MSE
                # --------------------------------------------

                loss = F.mse_loss(
                    noise_pred,
                    noise
                )

                accelerator.backward(
                    loss
                )

                # --------------------------------------------
                # Gradient clipping
                # --------------------------------------------

                accelerator.clip_grad_norm_(
                    model.parameters(),
                    1.0
                )

                # --------------------------------------------
                # Optimización
                # --------------------------------------------

                optimizer.step()

                lr_scheduler.step()

                optimizer.zero_grad()

            # ------------------------------------------------
            # Logging
            # ------------------------------------------------

            progress_bar.update(1)

            logs = {

                "loss":
                    loss.detach().item(),

                "lr":
                    lr_scheduler
                    .get_last_lr()[0],

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

        # ====================================================
        # EVALUACIÓN
        # ====================================================

        if accelerator.is_main_process:

            unwrapped_model = (
                accelerator.unwrap_model(
                    model
                )
            )

            if (
                (epoch + 1)
                % config.save_image_epochs
                == 0
                or epoch
                == config.num_epochs - 1
            ):

                evaluate(
                    config=config,
                    epoch=epoch,
                    model=unwrapped_model,
                    scheduler=noise_scheduler,
                    accelerator=accelerator,
                    dataset=dataset
                )

            # =================================================
            # GUARDADO
            # =================================================

            if (
                (epoch + 1)
                % config.save_model_epochs
                == 0
                or epoch
                == config.num_epochs - 1
            ):

                model_dir = os.path.join(
                    config.output_dir,
                    "transformer"
                )

                scheduler_dir = os.path.join(
                    config.output_dir,
                    "scheduler"
                )

                os.makedirs(
                    model_dir,
                    exist_ok=True
                )

                os.makedirs(
                    scheduler_dir,
                    exist_ok=True
                )

                # --------------------------------------------
                # Guardar Transformer
                # --------------------------------------------

                unwrapped_model.save_pretrained(
                    model_dir
                )

                # --------------------------------------------
                # Guardar scheduler
                # --------------------------------------------

                noise_scheduler.save_pretrained(
                    scheduler_dir
                )

                print(
                    f"\nModelo guardado en:"
                    f"\n{model_dir}"
                    f"\n\nScheduler guardado en:"
                    f"\n{scheduler_dir}\n"
                )

                # --------------------------------------------
                # Hugging Face
                # --------------------------------------------

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
        lr_scheduler,
        dataset
    )

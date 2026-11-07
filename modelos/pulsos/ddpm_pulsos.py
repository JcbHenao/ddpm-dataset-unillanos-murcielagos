import os
import math
from dataclasses import dataclass
from pathlib import Path
 
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
 
from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from huggingface_hub import create_repo, upload_folder
from tqdm.auto import tqdm

@dataclass
class TrainingConfig:
    image_size = 128
    train_batch_size = 40
    eval_batch_size = 16
    num_epochs = 50
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 10
    save_model_epochs = 30
    mixed_precision = 'fp16'
    output_dir = 'ddpm-pulsos-molossus-128-500ntt'
 
    push_to_hub = False
    hub_private_repo = False
    overwrite_output_dir = True
    seed = 42
 
 
config = TrainingConfig()
 

RUTA_DATASET_LOCAL = (
    "/media/adalab/bodega/CamiloBuitrago/"
    "Muestra_Molossus_Alta_Calidad_128"
)


class EspectrogramasDataset(Dataset):
    def __init__(self, carpeta, transform=None):
        self.carpeta = carpeta
        self.transform = transform
        self.archivos = sorted([
            f for f in os.listdir(carpeta)
            if f.lower().endswith(".png")
        ])
 
        if len(self.archivos) == 0:
            raise ValueError(
                f"No se encontraron imagenes .png en {carpeta}. "
            )
 
    def __len__(self):
        return len(self.archivos)
 
    def __getitem__(self, idx):
        ruta = os.path.join(self.carpeta, self.archivos[idx])
        imagen = Image.open(ruta).convert("RGB")
 
        if self.transform:
            imagen = self.transform(imagen)
 
        return {"images": imagen}
 
 
preprocess = transforms.Compose(
    [
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)
 
dataset = EspectrogramasDataset(RUTA_DATASET_LOCAL, transform=preprocess)
 
print(f"Imagenes cargadas para entrenamiento: {len(dataset)}")
 
train_dataloader = DataLoader(
    dataset,
    batch_size=config.train_batch_size,
    shuffle=True,
    num_workers=4,
)
 
 
# =====================================================
# Modelo UNet2D 
# =====================================================
# https://arxiv.org/pdf/2006.11239
 
model = UNet2DModel(
    sample_size=config.image_size,
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
 
noise_scheduler = DDPMScheduler(num_train_timesteps=500)
 
noise = torch.randn(sample_image.shape)
timesteps = torch.LongTensor([50])
noisy_image = noise_scheduler.add_noise(sample_image, noise, timesteps)
 
noise_pred = model(noisy_image, timesteps).sample
loss = F.mse_loss(noise_pred, noise)
 
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
 
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
)
 
 
# =====================================================
# Utilidades de evaluacion
# =====================================================
 
def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid
 
 
def evaluate(config, epoch, pipeline):
    images = pipeline(
        batch_size=config.eval_batch_size,
        generator=torch.manual_seed(config.seed),
    ).images
 
    image_grid = make_grid(images, rows=4, cols=4)
 
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")
 
 
# =====================================================
# Loop de entrenamiento 
# =====================================================
 
def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    logging_dir = os.path.join(config.output_dir, "logs")
    accelerator_project_config = ProjectConfiguration(
        project_dir=config.output_dir, logging_dir=logging_dir
    )
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_config=accelerator_project_config,
    )
    if accelerator.is_main_process:
        if config.push_to_hub:
            repo_id = create_repo(
                repo_id=Path(config.output_dir).name, exist_ok=True
            ).repo_id
        elif config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers("train_pulsos_molossus")
 
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
 
    global_step = 0
 
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(
            total=len(train_dataloader),
            disable=not accelerator.is_local_main_process,
        )
        progress_bar.set_description(f"Epoch {epoch}")
 
        for step, batch in enumerate(train_dataloader):
            clean_images = batch["images"]
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]
 
            timesteps = torch.randint(
                0, noise_scheduler.num_train_timesteps, (bs,),
                device=clean_images.device,
            ).long()
 
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
 
            with accelerator.accumulate(model):
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)
 
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
 
            progress_bar.update(1)
            logs = {
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "step": global_step,
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1
 
        if accelerator.is_main_process:
            pipeline = DDPMPipeline(
                unet=accelerator.unwrap_model(model), scheduler=noise_scheduler
            )
 
            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                evaluate(config, epoch, pipeline)
 
            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                if config.push_to_hub:
                    upload_folder(
                        repo_id=repo_id,
                        folder_path=config.output_dir,
                        commit_message=f"Epoch {epoch}",
                        ignore_patterns=["step_*", "epoch_*"],
                    )
                else:
                    pipeline.save_pretrained(config.output_dir)
 
 
if __name__ == "__main__":
    train_loop(
        config,
        model,
        noise_scheduler,
        optimizer,
        train_dataloader,
        lr_scheduler,
    )
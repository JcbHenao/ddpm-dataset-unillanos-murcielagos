import os
import json
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

# =====================================================
# VARIANTE CONDICIONADA POR CLASE
#   Un solo modelo para todas las especies. La especie entra a la UNet
#   como etiqueta de clase (class_labels), asi que al generar se le pide
#   la especie que se quiera.
#
# USO
#   Una GPU:      python train_ddpm_secuencias_condicional.py
#   Dos GPUs:     accelerate launch --multi_gpu --num_processes 2 train_ddpm_secuencias_condicional.py
#
#   Con dos GPUs cada una procesa su propio lote de train_batch_size, asi que
#   el lote efectivo es el doble (8) y cada epoca tiene la mitad de pasos.
#   Las metricas finales tambien se reparten: cada GPU evalua la mitad de las especies.
#
# Dependencias extra para las metricas:  pip install "torchmetrics[image]"
#
# PARA GENERAR DESPUES DEL ENTRENAMIENTO
#   from diffusers import UNet2DModel, DDPMScheduler
#   import train_ddpm_secuencias_condicional as t      # (o copiar la funcion generar)
#   unet = UNet2DModel.from_pretrained("ddpm-secuencias-1024x128-500ntt-condicional/unet").to("cuda")
#   sched = DDPMScheduler.from_pretrained("ddpm-secuencias-1024x128-500ntt-condicional/scheduler")
#   etiquetas = torch.tensor([t.ESPECIES.index("SBilineata")] * 8)
#   imagenes = t.a_pil(t.generar(unet, sched, etiquetas, torch.manual_seed(0)))
# =====================================================


@dataclass
class TrainingConfig:
    image_height = 128        # CAMBIO: antes image_size = 128 (cuadrado)
    image_width = 1024        
    train_batch_size = 4      # CAMBIO: antes 40; cada imagen tiene 8x mas pixeles que un pulso
    eval_batch_size = 16
    num_epochs = 50
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 10
    save_model_epochs = 30
    mixed_precision = 'fp16'  # si el loss vuelve a dar NaN, cambiar a 'no'
    output_dir = 'ddpm-secuencias-1024x128-500ntt-condicional'

    push_to_hub = False
    hub_private_repo = False
    overwrite_output_dir = True
    seed = 42

    # NUEVO: evaluacion final
    num_eval_samples = 500    # imagenes sinteticas POR ESPECIE para IS / FID / SSIM
    ssim_baseline_samples = 200   # reales por especie usadas para la linea base real-real


config = TrainingConfig()


RUTA_BASE = (
    "/media/adalab/bodega/CamiloBuitrago/espectrogramas/"
    "Espectrogramas_Secuencias_1024x128px"
)

# NUEVO: cada carpeta es una clase; la etiqueta numerica es la posicion en esta lista
ESPECIES = [
    "Molossus_1",
    "MolossusMolossus",
    "MolossusRufus",
    "Saccopteryx_sp",
    "SBilineata",
    "SLeptura",
]


class EspectrogramasDataset(Dataset):
    def __init__(self, carpeta_base, especies, transform=None):
        self.transform = transform
        self.archivos = []    # CAMBIO: lista de (ruta, etiqueta) con todas las especies
        for etiqueta, especie in enumerate(especies):
            carpeta = os.path.join(carpeta_base, especie)
            pngs = sorted([
                f for f in os.listdir(carpeta)
                if f.lower().endswith(".png")
            ])
            if len(pngs) == 0:
                raise ValueError(
                    f"No se encontraron imagenes .png en {carpeta}. "
                )
            print(f"  clase {etiqueta} = {especie}: {len(pngs)} imagenes")
            self.archivos += [(os.path.join(carpeta, f), etiqueta) for f in pngs]

    def __len__(self):
        return len(self.archivos)

    def __getitem__(self, idx):
        ruta, etiqueta = self.archivos[idx]
        imagen = Image.open(ruta).convert("RGB")

        if self.transform:
            imagen = self.transform(imagen)

        return {"images": imagen, "labels": etiqueta}   # CAMBIO: tambien devuelve la clase


preprocess = transforms.Compose(
    [
        transforms.Resize((config.image_height, config.image_width)),   # CAMBIO: (alto, ancho)
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

dataset = EspectrogramasDataset(RUTA_BASE, ESPECIES, transform=preprocess)

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
    sample_size=(config.image_height, config.image_width),   # CAMBIO: 128 x 1024
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
    num_class_embeds=len(ESPECIES),   
)


sample_image = dataset[0]["images"].unsqueeze(0)
sample_label = torch.LongTensor([dataset[0]["labels"]])
print("Input shape:", sample_image.shape)


# =====================================================
# Scheduler de ruido
# =====================================================

noise_scheduler = DDPMScheduler(num_train_timesteps=500)

noise = torch.randn(sample_image.shape)
timesteps = torch.LongTensor([50])
noisy_image = noise_scheduler.add_noise(sample_image, noise, timesteps)

noise_pred = model(noisy_image, timesteps, class_labels=sample_label).sample
loss = F.mse_loss(noise_pred, noise)

optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
)


# =====================================================
# NUEVO: muestreo condicionado. DDPMPipeline no acepta etiquetas de
# clase, asi que el bucle de eliminacion de ruido se hace aqui.
# =====================================================

@torch.no_grad()
def generar(unet, scheduler, etiquetas, generator=None):
    """Genera una imagen por etiqueta. Devuelve un arreglo (N, alto, ancho, 3) en [0, 1]."""
    device = next(unet.parameters()).device
    scheduler.set_timesteps(scheduler.config.num_train_timesteps)

    image = torch.randn(
        (len(etiquetas), 3, config.image_height, config.image_width),
        generator=generator,
    ).to(device)
    etiquetas = etiquetas.to(device)

    for t in tqdm(scheduler.timesteps, leave=False):
        model_output = unet(image, t, class_labels=etiquetas).sample
        image = scheduler.step(model_output, t, image, generator=generator).prev_sample

    image = (image / 2 + 0.5).clamp(0, 1)
    return image.cpu().permute(0, 2, 3, 1).numpy()


def a_pil(imagenes):
    return [Image.fromarray((img * 255).round().astype("uint8")) for img in imagenes]


# =====================================================
# Utilidades de evaluacion
# =====================================================

def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid


def evaluate(config, epoch, unet, scheduler):
    # CAMBIO: una fila por especie (en el orden de ESPECIES), 2 muestras por fila
    etiquetas = torch.arange(len(ESPECIES)).repeat_interleave(2)
    images = a_pil(generar(unet, scheduler, etiquetas, torch.manual_seed(config.seed)))

    image_grid = make_grid(images, rows=len(ESPECIES), cols=2)

    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")


# =====================================================
# NUEVO: metricas finales por especie (IS, FID, SSIM)
# =====================================================

def _ssim_contra_todas(img, banco, device, bs=64):
    """SSIM de una imagen (3,H,W uint8) contra cada imagen de banco (N,3,H,W uint8)."""
    from torchmetrics.functional.image import structural_similarity_index_measure as ssim

    a = img.to(device).float().div(255).unsqueeze(0)
    valores = []
    for i in range(0, len(banco), bs):
        b = banco[i:i + bs].to(device).float().div(255)
        v = ssim(a.expand(len(b), -1, -1, -1), b, data_range=1.0, reduction="none")
        valores.append(v.cpu())
    return torch.cat(valores)


@torch.no_grad()
def calcular_metricas(config, unet, scheduler, dataset, etiqueta):
    """Metricas de una especie: generadas con esa etiqueta contra las reales de esa especie."""
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.inception import InceptionScore

    especie = ESPECIES[etiqueta]
    device = next(unet.parameters()).device
    bs = config.eval_batch_size

    fid = FrechetInceptionDistance(feature=2048).to(device)
    inception = InceptionScore().to(device)

    # --- imagenes reales de esta especie (todas las de entrenamiento), en uint8 ---
    indices = [i for i, (_, e) in enumerate(dataset.archivos) if e == etiqueta]
    reales = torch.stack([
        ((dataset[i]["images"] * 0.5 + 0.5).clamp(0, 1) * 255).round().to(torch.uint8)
        for i in indices
    ])
    for i in range(0, len(reales), bs):
        fid.update(reales[i:i + bs].to(device), real=True)

    # --- imagenes generadas pidiendo esta especie ---
    dir_gen = os.path.join(config.output_dir, "generadas", especie)
    os.makedirs(dir_gen, exist_ok=True)
    generator = torch.Generator(device="cpu").manual_seed(config.seed + 1 + etiqueta)

    generadas = []
    n = 0
    while n < config.num_eval_samples:
        b = min(bs, config.num_eval_samples - n)
        imgs = generar(unet, scheduler, torch.full((b,), etiqueta), generator)
        for k, pil in enumerate(a_pil(imgs)):
            pil.save(os.path.join(dir_gen, f"gen_{n + k:05d}.png"))
        t = torch.from_numpy((imgs * 255).round().astype("uint8")).permute(0, 3, 1, 2)
        fid.update(t.to(device), real=False)
        inception.update(t.to(device))
        generadas.append(t)
        n += b
    generadas = torch.cat(generadas)

    is_media, is_std = inception.compute()

    # --- SSIM generada vs. reales ---
    ssim_medias, ssim_max = [], []
    for g in tqdm(generadas, desc=f"[{especie}] SSIM gen-real"):
        v = _ssim_contra_todas(g, reales, device)
        ssim_medias.append(v.mean())
        ssim_max.append(v.max())
    ssim_medias = torch.stack(ssim_medias)
    ssim_max = torch.stack(ssim_max)

    # --- linea base SSIM real vs. real (sin compararse consigo misma) ---
    g_base = torch.Generator().manual_seed(config.seed)
    idx = torch.randperm(len(reales), generator=g_base)[:config.ssim_baseline_samples]
    base_medias, base_max = [], []
    for j in tqdm(idx.tolist(), desc=f"[{especie}] SSIM real-real"):
        v = _ssim_contra_todas(reales[j], reales, device)
        v = torch.cat([v[:j], v[j + 1:]])
        base_medias.append(v.mean())
        base_max.append(v.max())

    return {
        "especie": especie,
        "n_reales": len(reales),
        "n_generadas": len(generadas),
        "IS": round(is_media.item(), 4),
        "IS_std": round(is_std.item(), 4),
        "FID": round(fid.compute().item(), 4),
        # promedio del SSIM de cada generada contra todas las reales
        "SSIM_medio_gen_real": round(ssim_medias.mean().item(), 4),
        # promedio del SSIM de cada generada contra su real mas parecida
        "SSIM_vecino_gen_real": round(ssim_max.mean().item(), 4),
        # referencias: cuanto se parecen las reales entre si
        "SSIM_medio_real_real": round(torch.stack(base_medias).mean().item(), 4),
        "SSIM_vecino_real_real": round(torch.stack(base_max).mean().item(), 4),
        # indicador de memorizacion: generadas casi identicas a una real
        "pct_gen_SSIM_mayor_0.9": round((ssim_max > 0.9).float().mean().item() * 100, 2),
    }


def guardar_modelo(config, unet, scheduler):
    # CAMBIO: se guardan UNet y scheduler por separado (no hay DDPMPipeline condicionado)
    unet.save_pretrained(os.path.join(config.output_dir, "unet"))
    scheduler.save_pretrained(os.path.join(config.output_dir, "scheduler"))
    with open(os.path.join(config.output_dir, "clases.json"), "w") as f:
        json.dump({i: e for i, e in enumerate(ESPECIES)}, f, indent=2)


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
        accelerator.init_trackers("train_secuencias_condicional")

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
            labels = batch["labels"]   # NUEVO: especie de cada imagen
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]

            timesteps = torch.randint(
                0, noise_scheduler.num_train_timesteps, (bs,),
                device=clean_images.device,
            ).long()

            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                noise_pred = model(
                    noisy_images, timesteps, class_labels=labels, return_dict=False
                )[0]   # CAMBIO: la UNet recibe la clase
                loss = F.mse_loss(noise_pred.float(), noise.float())   # CAMBIO: loss en float32

                # NUEVO: cortar de inmediato si el loss diverge, en vez de seguir entrenando con NaN
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"Loss no finito en epoca {epoch}, paso {global_step}. "
                        "Probar mixed_precision='no' o un learning_rate menor."
                    )

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

        unet = accelerator.unwrap_model(model)
        if accelerator.is_main_process:
            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                evaluate(config, epoch, unet, noise_scheduler)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                guardar_modelo(config, unet, noise_scheduler)
                if config.push_to_hub:
                    upload_folder(
                        repo_id=repo_id,
                        folder_path=config.output_dir,
                        commit_message=f"Epoch {epoch}",
                        ignore_patterns=["step_*", "epoch_*"],
                    )

    # NUEVO: metricas con el modelo final, una fila por especie.
    # Con varias GPUs cada proceso evalua las especies que le tocan y deja un json;
    # al final el proceso principal los junta en resumen_metricas.csv
    accelerator.wait_for_everyone()
    dir_metricas = os.path.join(config.output_dir, "metricas")
    os.makedirs(dir_metricas, exist_ok=True)
    for etiqueta in range(len(ESPECIES)):
        if etiqueta % accelerator.num_processes != accelerator.process_index:
            continue
        metricas = calcular_metricas(config, unet, noise_scheduler, train_dataloader.dataset, etiqueta)
        print(json.dumps(metricas, indent=2), flush=True)
        with open(os.path.join(dir_metricas, f"{ESPECIES[etiqueta]}.json"), "w") as f:
            json.dump(metricas, f, indent=2)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        resumen = []
        for especie in ESPECIES:
            with open(os.path.join(dir_metricas, f"{especie}.json")) as f:
                resumen.append(json.load(f))
        columnas = list(resumen[0].keys())
        with open(os.path.join(config.output_dir, "resumen_metricas.csv"), "w") as f:
            f.write(",".join(columnas) + "\n")
            for m in resumen:
                f.write(",".join(str(m[c]) for c in columnas) + "\n")
    accelerator.end_training()


if __name__ == "__main__":
    train_loop(
        config,
        model,
        noise_scheduler,
        optimizer,
        train_dataloader,
        lr_scheduler,
    )
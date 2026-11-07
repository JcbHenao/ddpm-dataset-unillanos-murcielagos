"""
Metricas finales por clase: IS, FID y SSIM (generada vs real y real vs real).
Misma logica que los scripts anteriores, para que los numeros sean comparables.
"""

import os

import torch
from tqdm.auto import tqdm

from .muestreo import a_pil, generar


def _a_uint8(tensor_normalizado):
    return ((tensor_normalizado * 0.5 + 0.5).clamp(0, 1) * 255).round().to(torch.uint8)


def _tres_canales(t):
    return t.repeat(1, 3, 1, 1) if t.shape[1] == 1 else t


def _ssim_contra_todas(img, banco, device, bs=64):
    from torchmetrics.functional.image import structural_similarity_index_measure as ssim

    a = img.to(device).float().div(255).unsqueeze(0)
    valores = []
    for i in range(0, len(banco), bs):
        b = banco[i:i + bs].to(device).float().div(255)
        v = ssim(a.expand(len(b), -1, -1, -1), b, data_range=1.0, reduction="none")
        valores.append(v.cpu())
    return torch.cat(valores)


@torch.no_grad()
def calcular(cfg, unet, scheduler_muestreo, dataset, etiqueta, carpeta_salida, condicional, etiquetas_fn=None):
    """
    etiquetas_fn(b, generator) -> tensor de etiquetas (multicondicional: se muestrean las
    combinaciones de condiciones de las imagenes reales de esa especie, para que la
    distribucion de n y patron de las generadas sea comparable con la de las reales).
    """
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.inception import InceptionScore

    d, r, ev = cfg["datos"], cfg["ruido"], cfg["evaluacion"]
    nombre_clase = dataset.clases[etiqueta]
    device = next(unet.parameters()).device
    bs = ev.get("lote", 16)
    n_muestras = ev.get("muestras_por_clase", 500)
    semilla = cfg["entrenamiento"].get("semilla", 42)

    fid = FrechetInceptionDistance(feature=2048).to(device)
    inception = InceptionScore().to(device)

    indices = dataset.indices_de_clase(etiqueta)
    reales = torch.stack([_a_uint8(dataset[i]["images"]) for i in indices])
    for i in range(0, len(reales), bs):
        fid.update(_tres_canales(reales[i:i + bs]).to(device), real=True)

    dir_gen = os.path.join(carpeta_salida, "generadas", nombre_clase)
    os.makedirs(dir_gen, exist_ok=True)
    generator = torch.Generator(device="cpu").manual_seed(semilla + 1 + etiqueta)

    generadas, n = [], 0
    barra = tqdm(total=n_muestras, desc=f"[{nombre_clase}] generando", leave=False)
    while n < n_muestras:
        b = min(bs, n_muestras - n)
        if etiquetas_fn is not None:
            etiquetas = etiquetas_fn(b, generator)
        else:
            etiquetas = torch.full((b,), etiqueta, dtype=torch.long) if condicional else None
        imgs = generar(unet, scheduler_muestreo, b, d["alto"], d["ancho"], d.get("canales", 3),
                       r.get("escala_entrada", 1.0), etiquetas, ev.get("pasos_muestreo"), generator,
                       guia=ev.get("guia", 1.0), precision=cfg["entrenamiento"].get("precision"))
        for k, pil in enumerate(a_pil(imgs)):
            pil.save(os.path.join(dir_gen, f"gen_{n + k:05d}.png"))
        t = torch.from_numpy((imgs * 255).round().astype("uint8")).permute(0, 3, 1, 2)
        fid.update(_tres_canales(t).to(device), real=False)
        inception.update(_tres_canales(t).to(device))
        generadas.append(t)
        n += b
        barra.update(b)
    barra.close()
    generadas = torch.cat(generadas)
    is_media, is_std = inception.compute()

    ssim_medias, ssim_max = [], []
    for g in tqdm(generadas, desc=f"[{nombre_clase}] SSIM gen-real", leave=False):
        v = _ssim_contra_todas(g, reales, device)
        ssim_medias.append(v.mean())
        ssim_max.append(v.max())
    ssim_medias, ssim_max = torch.stack(ssim_medias), torch.stack(ssim_max)

    g_base = torch.Generator().manual_seed(semilla)
    idx = torch.randperm(len(reales), generator=g_base)[:ev.get("ssim_linea_base", 200)]
    base_medias, base_max = [], []
    for j in tqdm(idx.tolist(), desc=f"[{nombre_clase}] SSIM real-real", leave=False):
        v = _ssim_contra_todas(reales[j], reales, device)
        v = torch.cat([v[:j], v[j + 1:]])
        if len(v):
            base_medias.append(v.mean())
            base_max.append(v.max())

    def _media(lista):
        return round(torch.stack(lista).mean().item(), 4) if lista else None

    return {
        "clase": nombre_clase,
        "n_reales": len(reales),
        "n_generadas": len(generadas),
        "IS": round(is_media.item(), 4),
        "IS_std": round(is_std.item(), 4),
        "FID": round(fid.compute().item(), 4),
        "SSIM_medio_gen_real": round(ssim_medias.mean().item(), 4),
        "SSIM_vecino_gen_real": round(ssim_max.mean().item(), 4),
        "SSIM_medio_real_real": _media(base_medias),
        "SSIM_vecino_real_real": _media(base_max),
        "pct_gen_SSIM_mayor_0.9": round((ssim_max > 0.9).float().mean().item() * 100, 2),
    }

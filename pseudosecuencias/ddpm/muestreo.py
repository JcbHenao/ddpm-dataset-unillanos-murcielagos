"""Muestreo (con o sin etiqueta de clase) y utilidades de imagen."""

from contextlib import contextmanager

import torch
from PIL import Image


@torch.no_grad()
def generar(unet, scheduler, n, alto, ancho, canales=3, escala_entrada=1.0,
            etiquetas=None, pasos=None, generator=None, guia=1.0, precision=None, descripcion=None):
    """
    Genera n imagenes. Devuelve un arreglo (n, alto, ancho, canales) en [0, 1].
    `scheduler` debe ser una copia de muestreo (ver ruido.scheduler_de_muestreo).
    guia > 1 (solo multicondicional): guia sin clasificador, mezcla la prediccion
    condicionada con la prediccion sin condiciones.
    precision "bf16"/"fp16": la UNet corre con autocast en GPU (~2-3 veces mas rapido
    que fp32); el paso del scheduler se hace siempre en fp32.
    descripcion: si se da, muestra una barra de progreso de los pasos de ruido.
    """
    usar_guia = guia > 1 and etiquetas is not None and hasattr(unet, "etiquetas_nulas")
    device = next(unet.parameters()).device
    scheduler.set_timesteps(pasos or scheduler.config.num_train_timesteps)

    x = torch.randn((n, canales, alto, ancho), generator=generator).to(device)
    x = x * scheduler.init_noise_sigma
    if etiquetas is not None:
        etiquetas = etiquetas.to(device)

    nulas = unet.etiquetas_nulas(n).to(device) if usar_guia else None
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(precision)
    usar_autocast = dtype is not None and device.type == "cuda"
    pasos_t = scheduler.timesteps
    if descripcion:
        from tqdm.auto import tqdm
        pasos_t = tqdm(pasos_t, desc=descripcion, leave=False)
    for t in pasos_t:
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=usar_autocast):
            salida = unet(x, t, class_labels=etiquetas).sample
            if usar_guia:
                sin_cond = unet(x, t, class_labels=nulas).sample
                salida = sin_cond + guia * (salida - sin_cond)
        x = scheduler.step(salida.float(), t, x, generator=generator).prev_sample

    x = (x / escala_entrada).clamp(-1, 1)
    return (x / 2 + 0.5).float().cpu().permute(0, 2, 3, 1).numpy()


def a_pil(imagenes):
    salida = []
    for img in imagenes:
        arr = (img * 255).round().astype("uint8")
        if arr.shape[-1] == 1:
            arr = arr[..., 0]
        salida.append(Image.fromarray(arr))
    return salida


def grilla(imagenes, filas, columnas):
    w, h = imagenes[0].size
    lienzo = Image.new(imagenes[0].mode, size=(columnas * w, filas * h))
    for i, im in enumerate(imagenes):
        lienzo.paste(im, box=(i % columnas * w, i // columnas * h))
    return lienzo


@contextmanager
def con_pesos_ema(ema, modelo):
    """Usa temporalmente los pesos EMA en `modelo` (si hay EMA)."""
    if ema is None:
        yield modelo
        return
    ema.store(modelo.parameters())
    ema.copy_to(modelo.parameters())
    try:
        yield modelo
    finally:
        ema.restore(modelo.parameters())

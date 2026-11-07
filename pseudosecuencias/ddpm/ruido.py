"""Construccion del scheduler de ruido y diagnostico del ruido en t = T."""

import math

from diffusers import DDIMScheduler, DDPMScheduler


def construir_scheduler(cfg_ruido):
    kwargs = dict(
        num_train_timesteps=cfg_ruido["pasos_entrenamiento"],
        beta_schedule=cfg_ruido["esquema"],
        prediction_type=cfg_ruido.get("prediccion", "epsilon"),
        rescale_betas_zero_snr=bool(cfg_ruido.get("snr_terminal_cero", False)),
        # "trailing": el muestreo arranca siempre en el ultimo paso (Lin et al. 2024)
        timestep_spacing="trailing",
        clip_sample=True,
    )
    if cfg_ruido["esquema"] == "linear":
        kwargs.update(beta_start=cfg_ruido.get("beta_start", 1e-4), beta_end=cfg_ruido.get("beta_end", 0.02))
    return DDPMScheduler(**kwargs)


def scheduler_de_muestreo(scheduler_entrenamiento, muestreador="ddpm"):
    """Copia independiente para muestrear (set_timesteps modifica el estado del scheduler)."""
    clase = DDIMScheduler if muestreador == "ddim" else DDPMScheduler
    return clase.from_config(scheduler_entrenamiento.config)


def diagnostico(scheduler, escala_entrada=1.0):
    """
    Senal que sobrevive en el ultimo paso de entrenamiento. Con escala_entrada b,
    la senal efectiva es b * sqrt(alpha_bar_T) y el SNR se multiplica por b^2.
    """
    ab_t = float(scheduler.alphas_cumprod[-1])
    senal = escala_entrada * math.sqrt(ab_t)
    snr = (escala_entrada ** 2) * ab_t / max(1.0 - ab_t, 1e-12)
    return {
        "sqrt_alpha_bar_T": math.sqrt(ab_t),
        "senal_efectiva_T": senal,
        "snr_T": snr,
        "porcentaje_senal_T": 100.0 * senal,
    }

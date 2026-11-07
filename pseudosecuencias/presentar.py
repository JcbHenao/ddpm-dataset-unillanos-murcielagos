"""
Genera (o rehace) las secuencias de prueba para mostrar de una corrida terminada,
con su modelo guardado. Es lo mismo que se hace al final de cada entrenamiento.

    python presentar.py runs/20_pseudo_sobreajuste
    python presentar.py runs/21_pseudo_multicond_lineal1500 --por-combinacion 8
    python presentar.py runs/21_pseudo_multicond_lineal1500 --metadatos datos/P4_dataset/metadatos.csv

Salida: runs/<corrida>/muestras_finales/ (hoja_<especie>.png, carpetas por combinacion, indice.csv).
--metadatos sirve si el dataset se movio despues de entrenar (las reales se toman de ahi).
"""

import argparse
import sys
from pathlib import Path

import torch

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))

from diffusers import DDPMScheduler, UNet2DModel  # noqa: E402

from ddpm import configuracion  # noqa: E402
from ddpm.datos import crear_dataset  # noqa: E402
from ddpm.modelo import UNetMulticondicional  # noqa: E402
from ddpm.presentacion import generar_muestras_finales  # noqa: E402
from ddpm.ruido import scheduler_de_muestreo  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir")
    parser.add_argument("--por-combinacion", type=int, default=None)
    parser.add_argument("--max-combinaciones", type=int, default=None)
    parser.add_argument("--metadatos", default=None, help="otra ruta de metadatos.csv (si se movio el dataset)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not (run_dir / "modelo").exists():
        sys.exit(f"{run_dir} no tiene modelo/: la corrida no ha terminado")
    cfg = configuracion.cargar(run_dir / "config.yaml")
    ev = cfg.setdefault("evaluacion", {})
    opciones = dict(ev.get("muestras_finales") or {"por_combinacion": 4})
    if args.por_combinacion is not None:
        opciones["por_combinacion"] = args.por_combinacion
    if args.max_combinaciones is not None:
        opciones["max_combinaciones"] = args.max_combinaciones
    ev["muestras_finales"] = opciones
    if args.metadatos:
        cfg["datos"]["metadatos"] = str(Path(args.metadatos).resolve())

    dataset = crear_dataset(cfg, semilla=cfg["entrenamiento"]["semilla"], permitir_redimensionar=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if cfg["modo"] == "multicondicional":
        unet = UNetMulticondicional.cargar(run_dir / "modelo")
        if {k: list(v) for k, v in unet.vocab.items()} != {k: list(v) for k, v in dataset.vocab.items()}:
            sys.exit(f"Las condiciones del modelo {unet.vocab} no coinciden con las del dataset {dataset.vocab}: "
                     "revisar --metadatos (debe ser el dataset con el que se entreno)")
    else:
        unet = UNet2DModel.from_pretrained(run_dir / "modelo" / "unet")
    unet = unet.to(device).eval()
    scheduler = DDPMScheduler.from_pretrained(run_dir / "modelo" / "scheduler")
    scheduler = scheduler_de_muestreo(scheduler, ev.get("muestreador", "ddpm"))
    generar_muestras_finales(cfg, unet, scheduler, dataset, run_dir / "muestras_finales")


if __name__ == "__main__":
    main()

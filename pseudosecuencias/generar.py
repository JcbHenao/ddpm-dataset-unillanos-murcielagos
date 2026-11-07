"""
Genera imagenes con el modelo de una corrida terminada (pesos EMA si los hubo).

    python generar.py runs/02_secuencias_cond_lineal1500 --especie SBilineata --cantidad 100
    python generar.py runs/04_secuencias_por_especie_lineal1500-SLeptura --cantidad 50 --ddim 100

  Modelos multicondicionales (pseudo-secuencias): cada condicion es opcional;
  la que no se indique queda "sin dato" y el modelo la elige.
    python generar.py runs/21_pseudo_multicond_lineal1500 --especie MolossusMolossus \
        --pulsos 3 --patron alto-bajo-alto --cantidad 32 --guia 2

Las imagenes quedan en runs/<corrida>/generadas_extra/<especie>/.
"""

import argparse
import sys
from pathlib import Path

import torch

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))

from diffusers import DDPMScheduler, UNet2DModel  # noqa: E402

from ddpm import configuracion, registro  # noqa: E402
from ddpm.modelo import UNetMulticondicional  # noqa: E402
from ddpm.muestreo import a_pil, generar  # noqa: E402
from ddpm.ruido import scheduler_de_muestreo  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir")
    parser.add_argument("--especie", help="obligatoria si el modelo es condicional; opcional si es multicondicional")
    parser.add_argument("--pulsos", help="multicondicional: numero de pulsos (2 o 3)")
    parser.add_argument("--patron", help="multicondicional: p. ej. alto-bajo, bajo-alto, alto-bajo-alto, bajo-alto-bajo")
    parser.add_argument("--guia", type=float, default=1.0, help="multicondicional: guia sin clasificador (1 = sin guia)")
    parser.add_argument("--cantidad", type=int, default=16, help="imagenes a generar")
    parser.add_argument("--lote", type=int, default=16)
    parser.add_argument("--ddim", type=int, default=None, help="usar DDIM con N pasos en vez de DDPM completo")
    parser.add_argument("--semilla", type=int, default=0)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    cfg = configuracion.cargar(run_dir / "config.yaml")
    d, r = cfg["datos"], cfg["ruido"]
    clases = registro.leer_json(run_dir / "modelo" / "clases.json")
    clases = [clases[str(i)] for i in range(len(clases))]
    condicional = cfg["modo"] == "condicional"
    multi = cfg["modo"] == "multicondicional"

    if multi:
        nombre = "_".join(str(v) for v in (args.especie, f"n{args.pulsos}" if args.pulsos else None, args.patron)
                          if v) or "sin_condicion"
    elif condicional:
        if args.especie not in clases:
            sys.exit(f"--especie debe ser una de {clases}")
        etiqueta = clases.index(args.especie)
        nombre = args.especie
    else:
        nombre = clases[0]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if multi:
        unet = UNetMulticondicional.cargar(run_dir / "modelo").to(device).eval()
        valores = {"especie": args.especie, "n": args.pulsos, "patron": args.patron}
        etiqueta_multi = unet.codificar({k: v for k, v in valores.items() if k in unet.nombres})
        print(f"Condiciones: {unet.describir(etiqueta_multi[0])}")
    else:
        unet = UNet2DModel.from_pretrained(run_dir / "modelo" / "unet").to(device).eval()
    scheduler = DDPMScheduler.from_pretrained(run_dir / "modelo" / "scheduler")
    scheduler = scheduler_de_muestreo(scheduler, "ddim" if args.ddim else "ddpm")

    salida = run_dir / "generadas_extra" / nombre
    salida.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(args.semilla)
    hechas = 0
    while hechas < args.cantidad:
        b = min(args.lote, args.cantidad - hechas)
        if multi:
            etiquetas = etiqueta_multi.repeat(b, 1)
        else:
            etiquetas = torch.full((b,), etiqueta, dtype=torch.long) if condicional else None
        imgs = generar(unet, scheduler, b, d["alto"], d["ancho"], d.get("canales", 3),
                       r.get("escala_entrada", 1.0), etiquetas, args.ddim, generator, guia=args.guia,
                       precision=cfg["entrenamiento"].get("precision"), descripcion="pasos de ruido")
        for k, im in enumerate(a_pil(imgs)):
            im.save(salida / f"gen_s{args.semilla}_{hechas + k:05d}.png")
        hechas += b
        print(f"{hechas}/{args.cantidad}")
    print(f"Listo: {salida}")


if __name__ == "__main__":
    main()

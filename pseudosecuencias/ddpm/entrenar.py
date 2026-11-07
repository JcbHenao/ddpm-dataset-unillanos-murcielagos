"""
Entrena UNA corrida. Normalmente lo llama lanzar.py, pero tambien se puede usar directo:

    python -m ddpm.entrenar --run-dir runs/<nombre>

La carpeta debe tener config.yaml (lanzar.py la crea). Si existe
checkpoints/ultimo, la corrida se reanuda desde ahi.
"""

import argparse
import csv
import math
import time
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.training_utils import EMAModel
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from . import configuracion, metricas, registro
from .datos import crear_dataset
from .modelo import UNetMulticondicional, construir_modelo, contar_parametros
from .muestreo import a_pil, con_pesos_ema, generar, grilla
from .presentacion import generar_muestras_finales
from .ruido import construir_scheduler, diagnostico, scheduler_de_muestreo


def _guardar_grilla(cfg, unet, scheduler_muestreo, dataset, ruta):
    d, r, ev = cfg["datos"], cfg["ruido"], cfg.get("evaluacion", {})
    clases = dataset.clases
    if isinstance(unet, UNetMulticondicional):
        # una fila por cada una de las combinaciones de condiciones mas frecuentes, 2 muestras por fila
        combinaciones = dataset.combinaciones_frecuentes(ev.get("filas_grilla", 12))
        etiquetas = combinaciones.repeat_interleave(2, dim=0)
        n, filas, columnas = len(etiquetas), len(combinaciones), 2
        Path(ruta).with_suffix(".txt").write_text(
            "\n".join(f"fila {i + 1}: {unet.describir(c)}" for i, c in enumerate(combinaciones)) + "\n",
            encoding="utf-8")
    elif cfg["modo"] == "condicional":
        etiquetas = torch.arange(len(clases)).repeat_interleave(2)
        n, filas, columnas = len(etiquetas), len(clases), 2
    else:
        etiquetas = None
        columnas = 2 if d["ancho"] >= 2 * d["alto"] else 4
        n, filas = 8, 8 // columnas
    generator = torch.Generator().manual_seed(cfg["entrenamiento"]["semilla"])
    pasos = ev.get("pasos_muestreo") or r["pasos_entrenamiento"]
    print(f"[{cfg['nombre']}] generando grilla {Path(ruta).name}: {n} imagenes, "
          f"{ev.get('muestreador', 'ddpm')} {pasos} pasos", flush=True)
    imgs = generar(unet, scheduler_muestreo, n, d["alto"], d["ancho"], d.get("canales", 3),
                   r.get("escala_entrada", 1.0), etiquetas, ev.get("pasos_muestreo"), generator,
                   guia=ev.get("guia", 1.0), precision=cfg["entrenamiento"].get("precision"),
                   descripcion=f"[{cfg['nombre']}] grilla")
    grilla(a_pil(imgs), filas, columnas).save(ruta)
    print(f"[{cfg['nombre']}] grilla guardada en {ruta}", flush=True)


def entrenar(run_dir):
    run_dir = Path(run_dir).resolve()
    cfg = configuracion.cargar(run_dir / "config.yaml")
    configuracion.validar_o_fallar(cfg, str(run_dir))
    d, r, e, ev = cfg["datos"], cfg["ruido"], cfg["entrenamiento"], cfg.get("evaluacion", {})

    accelerator = Accelerator(
        mixed_precision=e["precision"],
        gradient_accumulation_steps=e.get("acumulacion", 1),
        log_with="tensorboard",
        project_config=ProjectConfiguration(project_dir=str(run_dir), logging_dir=str(run_dir / "logs")),
    )
    principal = accelerator.is_main_process
    set_seed(e["semilla"])
    condicional = cfg["modo"] == "condicional"
    multi = cfg["modo"] == "multicondicional"
    usa_etiquetas = condicional or multi
    escala = r.get("escala_entrada", 1.0)
    t_inicio = time.time()

    try:
        if accelerator.device.type == "cpu" and not e.get("permitir_cpu", False):
            raise RuntimeError(
                "No hay GPU disponible para PyTorch: el entrenamiento correria en CPU (dias). "
                f"PyTorch {torch.__version__} esta compilado para CUDA {torch.version.cuda}; si el aviso dice "
                "'NVIDIA driver ... too old', reinstalar PyTorch para la version de CUDA del driver "
                "(ver 'nvidia-smi' y el README). Para entrenar en CPU a proposito: entrenamiento.permitir_cpu: true"
            )
        # ---------------- datos, modelo, ruido, optimizador ----------------
        dataset = crear_dataset(cfg, semilla=e["semilla"])
        for aviso in getattr(dataset, "avisos", []):
            accelerator.print(f"[AVISO] {aviso}")
        n_workers = d.get("num_workers", 4)
        loader = DataLoader(dataset, batch_size=e["lote"], shuffle=True, num_workers=n_workers,
                            pin_memory=torch.cuda.is_available(), persistent_workers=n_workers > 0)

        unet = construir_modelo(cfg, dataset)   # UNet, UNet con clase o UNetMulticondicional
        scheduler = construir_scheduler(r)

        lotes_por_proceso = math.ceil(len(dataset) / (e["lote"] * accelerator.num_processes))
        pasos_por_epoca = math.ceil(lotes_por_proceso / e.get("acumulacion", 1))
        pasos_totales = pasos_por_epoca * e["epocas"]
        if e.get("pasos_max"):
            pasos_totales = min(pasos_totales, e["pasos_max"])

        optimizador = torch.optim.AdamW(unet.parameters(), lr=e["lr"])
        # accelerate avanza el scheduler de lr una vez por proceso en cada paso de optimizador
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizador,
            num_warmup_steps=e["warmup"] * accelerator.num_processes,
            num_training_steps=pasos_totales * accelerator.num_processes,
        )

        ema = None
        if e.get("ema", {}).get("usar", False):
            extra = {} if multi else {"model_cls": UNet2DModel, "model_config": unet.config}
            ema = EMAModel(unet.parameters(), decay=e["ema"].get("decay", 0.9999),
                           use_ema_warmup=e["ema"].get("warmup", True), power=e["ema"].get("power", 0.75),
                           **extra)

        unet, optimizador, loader, lr_scheduler = accelerator.prepare(unet, optimizador, loader, lr_scheduler)
        if ema is not None:
            ema.to(accelerator.device)
            accelerator.register_for_checkpointing(ema)

        # ---------------- reanudar ----------------
        dir_checkpoints = run_dir / "checkpoints"
        dir_ultimo = dir_checkpoints / "ultimo"
        progreso = registro.leer_json(dir_checkpoints / "progreso.json", {}) or {}
        epoca_inicial, paso_global = 0, 0
        if dir_ultimo.is_dir() and progreso:
            accelerator.load_state(str(dir_ultimo))
            epoca_inicial, paso_global = progreso["epoca"] + 1, progreso["paso_global"]
            accelerator.print(f"Reanudando desde la epoca {epoca_inicial} (paso {paso_global})")

        diag = diagnostico(scheduler, escala)
        if principal:
            accelerator.init_trackers("entrenamiento")
            (run_dir / "muestras").mkdir(exist_ok=True)
            registro.actualizar_estado(
                run_dir, estado="en_curso", inicio_entrenamiento=registro.ahora(),
                imagenes=len(dataset), clases=dataset.conteo_por_clase(),
                parametros_unet=contar_parametros(accelerator.unwrap_model(unet)),
                gpus=accelerator.num_processes,
                lote_efectivo=e["lote"] * e.get("acumulacion", 1) * accelerator.num_processes,
                pasos_por_epoca=pasos_por_epoca, pasos_totales=pasos_totales, diagnostico_ruido=diag,
                condiciones=dataset.vocab if multi else None,
            )
            print(f"\n=== {cfg['nombre']} ===")
            print(f"Imagenes: {len(dataset)}  por clase: {dataset.conteo_por_clase()}")
            if multi:
                print(f"Condiciones: {dataset.vocab}")
                print(f"Combinaciones distintas en entrenamiento: {len(dataset.conteo_combinaciones())}")
            print(f"Pasos de optimizador: {pasos_por_epoca}/epoca, {pasos_totales} en total")
            print(f"Senal que queda en t=T: {diag['porcentaje_senal_T']:.4f} %")

        scheduler_muestreo = scheduler_de_muestreo(scheduler, ev.get("muestreador", "ddpm"))
        ruta_perdidas = run_dir / "perdidas_por_epoca.csv"
        terminado = paso_global >= pasos_totales

        # ---------------- bucle de entrenamiento ----------------
        for epoca in range(epoca_inicial, e["epocas"]):
            if terminado:
                break
            unet.train()
            suma, n_lotes, t_epoca = 0.0, 0, time.time()
            barra = tqdm(total=len(loader), disable=not accelerator.is_local_main_process,
                         desc=f"[{cfg['nombre']}] epoca {epoca}")

            for lote in loader:
                x0 = lote["images"] * escala
                etiquetas = lote["labels"] if usa_etiquetas else None
                ruido = torch.randn_like(x0)
                t = torch.randint(0, scheduler.config.num_train_timesteps, (x0.shape[0],), device=x0.device).long()
                xt = scheduler.add_noise(x0, ruido, t)
                objetivo = ruido if r["prediccion"] == "epsilon" else scheduler.get_velocity(x0, ruido, t)

                with accelerator.accumulate(unet):
                    pred = unet(xt, t, class_labels=etiquetas, return_dict=False)[0]
                    perdida = F.mse_loss(pred.float(), objetivo.float())
                    if not torch.isfinite(perdida):
                        raise RuntimeError(
                            f"Perdida no finita en epoca {epoca}, paso {paso_global}. "
                            "Probar precision 'bf16' o 'no', o un lr menor."
                        )
                    accelerator.backward(perdida)
                    if accelerator.sync_gradients and e.get("clip_grad"):
                        accelerator.clip_grad_norm_(unet.parameters(), e["clip_grad"])
                    optimizador.step()
                    lr_scheduler.step()
                    optimizador.zero_grad()

                if accelerator.sync_gradients:
                    if ema is not None:
                        ema.step(unet.parameters())
                    paso_global += 1
                    accelerator.log({"perdida": perdida.detach().item(),
                                     "lr": lr_scheduler.get_last_lr()[0]}, step=paso_global)

                suma += perdida.detach().item()
                n_lotes += 1
                barra.update(1)
                barra.set_postfix(perdida=f"{perdida.detach().item():.4f}", paso=paso_global)
                if paso_global >= pasos_totales:
                    terminado = True
                    break
            barra.close()

            ultima = terminado or epoca == e["epocas"] - 1
            if principal:
                nuevo = not ruta_perdidas.exists()
                with open(ruta_perdidas, "a", newline="") as f:
                    w = csv.writer(f)
                    if nuevo:
                        w.writerow(["epoca", "paso_global", "perdida_media", "lr", "segundos_epoca"])
                    w.writerow([epoca, paso_global, round(suma / max(n_lotes, 1), 6),
                                lr_scheduler.get_last_lr()[0], round(time.time() - t_epoca, 1)])

                if (epoca + 1) % e.get("cada_muestras_epocas", 10) == 0 or ultima:
                    modelo = accelerator.unwrap_model(unet)
                    modelo.eval()
                    with con_pesos_ema(ema, modelo):
                        _guardar_grilla(cfg, modelo, scheduler_muestreo, dataset,
                                        run_dir / "muestras" / f"epoca_{epoca:04d}.png")
                    modelo.train()

                if (epoca + 1) % e.get("cada_checkpoint_epocas", 10) == 0 or ultima:
                    accelerator.save_state(str(dir_ultimo))
                    registro.escribir_json(dir_checkpoints / "progreso.json",
                                           {"epoca": epoca, "paso_global": paso_global, "fecha": registro.ahora()})
                    registro.actualizar_estado(run_dir, epoca=epoca, paso_global=paso_global)
            accelerator.wait_for_everyone()

        # ---------------- guardar modelo final (pesos EMA si hay) ----------------
        modelo = accelerator.unwrap_model(unet)
        modelo.eval()
        if principal:
            dir_modelo = run_dir / "modelo"
            with con_pesos_ema(ema, modelo):
                if multi:
                    modelo.guardar(dir_modelo)    # unet/ + condicionador.pt + condiciones.json
                else:
                    modelo.save_pretrained(dir_modelo / "unet")
            scheduler.save_pretrained(dir_modelo / "scheduler")
            registro.escribir_json(dir_modelo / "clases.json", dict(enumerate(dataset.clases)))
            # secuencias de prueba para mostrar: generadas vs reales por combinacion de condiciones
            with con_pesos_ema(ema, modelo):
                generar_muestras_finales(cfg, modelo, scheduler_muestreo, dataset, run_dir / "muestras_finales")
        accelerator.wait_for_everyone()

        # ---------------- metricas (cada GPU evalua una parte de las clases) ----------------
        if ev.get("calcular_al_final", True):
            dir_metricas = run_dir / "metricas"
            if principal:
                dir_metricas.mkdir(exist_ok=True)
            accelerator.wait_for_everyone()
            if ema is not None:
                ema.copy_to(modelo.parameters())
            for etiqueta in range(len(dataset.clases)):
                if etiqueta % accelerator.num_processes != accelerator.process_index:
                    continue
                etiquetas_fn = None
                if multi:
                    # las generadas de esta especie reciben las combinaciones (n, patron, ...)
                    # de sus imagenes reales, muestreadas con reemplazo
                    reales = dataset.etiquetas_de_clase(etiqueta)
                    etiquetas_fn = (lambda b, g, reales=reales:
                                    reales[torch.randint(len(reales), (b,), generator=g)])
                res = metricas.calcular(cfg, modelo, scheduler_de_muestreo(scheduler, ev.get("muestreador", "ddpm")),
                                        dataset, etiqueta, str(run_dir), condicional, etiquetas_fn)
                print(res, flush=True)
                registro.escribir_json(dir_metricas / f"{dataset.clases[etiqueta]}.json", res)
            accelerator.wait_for_everyone()
            if principal:
                filas = [registro.leer_json(dir_metricas / f"{c}.json") for c in dataset.clases]
                filas = [f for f in filas if f]
                registro.escribir_json(run_dir / "metricas.json", filas)
                if filas:
                    with open(run_dir / "metricas.csv", "w", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
                        w.writeheader()
                        w.writerows(filas)

        if principal:
            registro.actualizar_estado(run_dir, estado="completado", fin=registro.ahora(),
                                       duracion_horas=round((time.time() - t_inicio) / 3600, 3),
                                       paso_global=paso_global)
        accelerator.end_training()

    except KeyboardInterrupt:
        if principal:
            registro.actualizar_estado(run_dir, estado="interrumpido", fin=registro.ahora())
        raise
    except Exception:
        if principal:
            registro.actualizar_estado(run_dir, estado="fallido", fin=registro.ahora(),
                                       error=traceback.format_exc()[-3000:])
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True, help="carpeta de la corrida (con config.yaml)")
    args = parser.parse_args()
    entrenar(args.run_dir)


if __name__ == "__main__":
    main()

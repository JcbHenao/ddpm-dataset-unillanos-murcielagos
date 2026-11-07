"""
Verifica que el entorno y los datos esten listos ANTES de lanzar entrenamientos.

    python verificar_entorno.py                                   # solo entorno
    python verificar_entorno.py configs/experimentos/             # entorno + cada experimento
    python verificar_entorno.py configs/experimentos/02_*.yaml --sin-prueba-gpu

Revisa:
  1. Python, PyTorch, CUDA, GPUs, soporte bf16 y versiones de paquetes.
  2. Por experimento: configuracion valida, carpetas de datos, imagenes por
     especie, tamano real de las imagenes contra alto x ancho configurados.
  3. Ruido: cuanta senal queda en el ultimo paso (deberia ser ~0).
  4. Prueba real: un paso de entrenamiento con un lote de datos reales en la GPU,
     memoria pico y estimacion de duracion de la corrida.

El reporte queda tambien en verificacion_entorno.txt.
"""

import argparse
import math
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))

LINEAS = []
CONTEO = {"OK": 0, "AVISO": 0, "ERROR": 0}


def informar(nivel, texto):
    CONTEO[nivel] = CONTEO.get(nivel, 0) + 1
    linea = f"[{nivel:<5}] {texto}"
    LINEAS.append(linea)
    print(linea, flush=True)


def titulo(texto):
    linea = f"\n=== {texto} ==="
    LINEAS.append(linea)
    print(linea, flush=True)


def verificar_paquetes():
    titulo("Entorno")
    informar("OK", f"Python {sys.version.split()[0]} en {sys.executable}")
    requeridos = ["torch", "torchvision", "diffusers", "accelerate", "torchmetrics", "torch-fidelity",
                  "tensorboard", "pyyaml", "pillow", "numpy", "tqdm", "scipy"]
    from importlib.metadata import PackageNotFoundError, version
    faltan = []
    for p in requeridos:
        try:
            informar("OK", f"{p} {version(p)}")
        except PackageNotFoundError:
            faltan.append(p)
            informar("ERROR", f"falta el paquete {p} (pip install -r requirements.txt)")
    if faltan:
        return False

    import torch
    if torch.cuda.is_available():
        informar("OK", f"CUDA {torch.version.cuda}, {torch.cuda.device_count()} GPU(s)")
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            informar("OK", f"  GPU {i}: {p.name}, {p.total_memory / 1e9:.1f} GB")
        if torch.cuda.is_bf16_supported():
            informar("OK", "bf16 soportado (recomendado sobre fp16)")
        else:
            informar("AVISO", "bf16 NO soportado en esta GPU: usar precision fp16 o no")
    else:
        driver = None
        try:
            import re
            import subprocess
            salida = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10).stdout
            m = re.search(r"CUDA Version:\s*([\d.]+)", salida)
            driver = m.group(1) if m else None
        except Exception:
            pass
        if driver:
            informar("ERROR", f"PyTorch no ve la GPU: esta compilado para CUDA {torch.version.cuda} y el driver "
                              f"solo soporta hasta CUDA {driver}. Reinstalar PyTorch para CUDA {driver} o "
                              f"menor (ver README, seccion Instalacion)")
        else:
            informar("ERROR", "CUDA no disponible (nvidia-smi no responde): el entrenamiento correria en CPU. "
                              "Si es intencional, poner entrenamiento.permitir_cpu: true")
    return True


def verificar_experimento(cfg, prueba_gpu):
    from ddpm import configuracion
    from ddpm.datos import crear_dataset
    from ddpm.ruido import construir_scheduler, diagnostico

    titulo(f"Experimento: {cfg['nombre']}")
    errores = configuracion.validar(cfg)
    for err in errores:
        informar("ERROR", err)
    if errores:
        return

    d, r, e = cfg["datos"], cfg["ruido"], cfg["entrenamiento"]
    informar("OK", f"modo={cfg['modo']}  resolucion={d['alto']}x{d['ancho']} (alto x ancho)  canales={d['canales']}")

    # --- datos ---
    try:
        ds = crear_dataset(cfg, semilla=e["semilla"], permitir_redimensionar=True)  # para reportar tamanos
    except Exception as ex:
        informar("ERROR", f"datos: {ex}")
        return
    for aviso in getattr(ds, "avisos", []):
        informar("AVISO", aviso)
    for clase, n in ds.conteo_por_clase().items():
        nivel = "AVISO" if n < 100 else "OK"
        extra = " (pocas imagenes: riesgo de memorizacion)" if n < 100 else ""
        informar(nivel, f"  {clase}: {n} imagenes{extra}")
    if cfg["modo"] == "multicondicional":
        informar("OK", f"condiciones: {ds.vocab}")
        combos = ds.conteo_combinaciones()
        pocas = {k: v for k, v in combos.items() if v < 20}
        informar("OK", f"{len(combos)} combinaciones distintas de condiciones en entrenamiento")
        if pocas:
            informar("AVISO", f"{len(pocas)} combinaciones con menos de 20 imagenes: se apoyan en lo que "
                              f"comparten con otras (embeddings separados por condicion)")
    esperado = (d["ancho"], d["alto"])
    for (ancho, alto), n in sorted(ds.tamanos.items()):
        if (ancho, alto) == esperado:
            informar("OK", f"  {n} imagenes de {alto}x{ancho} px (alto x ancho, coincide con la configuracion)")
        else:
            nivel = "AVISO" if d.get("permitir_redimensionar") else "ERROR"
            informar(nivel, f"  {n} imagenes de {alto}x{ancho} px (alto x ancho), pero la configuracion pide "
                            f"{d['alto']}x{d['ancho']} (se redimensionarian, cambiando la escala de tiempo/frecuencia)")

    # --- ruido ---
    scheduler = construir_scheduler(r)
    diag = diagnostico(scheduler, r.get("escala_entrada", 1.0))
    pct = diag["porcentaje_senal_T"]
    nivel = "OK" if pct < 1.0 else "AVISO"
    consejo = "" if pct < 1.0 else " -> el modelo nunca ve ruido puro al entrenar; subir pasos o usar coseno/SNR cero"
    informar(nivel, f"senal que queda en t=T: {pct:.4f} % (esquema {r['esquema']}, "
                    f"{r['pasos_entrenamiento']} pasos, escala {r.get('escala_entrada', 1.0)}){consejo}")

    lotes = math.ceil(len(ds) / e["lote"])
    pasos_epoca = math.ceil(lotes / e.get("acumulacion", 1))
    informar("OK", f"lote efectivo {e['lote'] * e.get('acumulacion', 1)} por GPU -> {pasos_epoca} pasos/epoca, "
                   f"{pasos_epoca * e['epocas']} pasos en {e['epocas']} epocas")
    if pasos_epoca * e["epocas"] < 10000:
        informar("AVISO", "menos de 10 000 pasos de optimizador en total: suele ser poco para un DDPM")

    if prueba_gpu:
        try:
            prueba_de_paso(cfg, ds, scheduler, lotes)
        except Exception as ex:
            informar("ERROR", f"fallo el paso de prueba: {type(ex).__name__}: {ex}")


def prueba_de_paso(cfg, ds, scheduler, lotes_por_epoca):
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    from ddpm.modelo import construir_modelo, contar_parametros

    d, r, e = cfg["datos"], cfg["ruido"], cfg["entrenamiento"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    condicional = cfg["modo"] in ("condicional", "multicondicional")
    lote = e["lote"] if device == "cuda" else 1
    unet = construir_modelo(cfg, ds).to(device)
    informar("OK", f"UNet: {contar_parametros(unet) / 1e6:.1f} M parametros")
    opt = torch.optim.AdamW(unet.parameters(), lr=e["lr"])
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(e["precision"])
    usar_autocast = dtype is not None and device == "cuda"
    escalador = torch.amp.GradScaler("cuda", enabled=(e["precision"] == "fp16" and device == "cuda"))

    batch = next(iter(DataLoader(ds, batch_size=lote, shuffle=True)))
    x0 = batch["images"].to(device) * r.get("escala_entrada", 1.0)
    etiquetas = batch["labels"].to(device) if condicional else None
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    tiempos = []
    try:
        for i in range(4):
            t0 = time.time()
            ruido = torch.randn_like(x0)
            t = torch.randint(0, scheduler.config.num_train_timesteps, (x0.shape[0],), device=device)
            xt = scheduler.add_noise(x0, ruido, t)
            objetivo = ruido if r["prediccion"] == "epsilon" else scheduler.get_velocity(x0, ruido, t)
            with torch.autocast(device_type="cuda", dtype=dtype, enabled=usar_autocast):
                pred = unet(xt, t, class_labels=etiquetas).sample
            perdida = F.mse_loss(pred.float(), objetivo.float())
            escalador.scale(perdida).backward()
            escalador.step(opt)
            escalador.update()
            opt.zero_grad()
            if device == "cuda":
                torch.cuda.synchronize()
            if i > 0:
                tiempos.append(time.time() - t0)
            if device == "cpu":
                break
    except torch.cuda.OutOfMemoryError:
        informar("ERROR", f"sin memoria con lote {lote}: bajar entrenamiento.lote y subir acumulacion")
        return
    if not torch.isfinite(perdida):
        informar("ERROR", f"la perdida dio {perdida.item()} en la prueba ({e['precision']})")
        return
    informar("OK", f"paso de prueba en {device} con lote {lote}: perdida {perdida.item():.4f}")
    if device == "cuda":
        pico = torch.cuda.max_memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        informar("OK" if pico < 0.9 * total else "AVISO", f"memoria pico {pico:.1f} de {total:.1f} GB")
        seg = sum(tiempos) / len(tiempos)
        horas = seg * lotes_por_epoca * e["epocas"] / 3600
        informar("OK", f"{seg:.2f} s por lote -> ~{horas:.1f} h de entrenamiento con 1 GPU "
                       f"(sin contar muestras ni metricas)")
        estimar_muestreo(cfg, ds, unet, x0, etiquetas, dtype, usar_autocast)
    else:
        informar("AVISO", "prueba hecha en CPU con lote 1; no sirve para estimar memoria ni tiempo")


def estimar_muestreo(cfg, ds, unet, x0, etiquetas, dtype, usar_autocast):
    """Mide un paso de muestreo y estima cuanto tardan las grillas y las metricas."""
    import torch

    r, e, ev = cfg["ruido"], cfg["entrenamiento"], cfg.get("evaluacion", {})
    lote = ev.get("lote", 16)
    reps = math.ceil(lote / x0.shape[0])
    x = x0.repeat(reps, 1, 1, 1)[:lote]
    etq = etiquetas.repeat(reps, *([1] * (etiquetas.dim() - 1)))[:lote] if etiquetas is not None else None
    t = torch.full((lote,), r["pasos_entrenamiento"] // 2, device=x.device)
    unet.eval()
    tiempos = []
    try:
        with torch.no_grad():
            for i in range(4):
                t0 = time.time()
                with torch.autocast(device_type=x.device.type, dtype=dtype, enabled=usar_autocast):
                    unet(x, t, class_labels=etq)
                if x.device.type == "cuda":
                    torch.cuda.synchronize()
                if i > 0:
                    tiempos.append(time.time() - t0)
    except torch.cuda.OutOfMemoryError:
        informar("AVISO", f"sin memoria al muestrear con evaluacion.lote {lote}: bajarlo")
        return
    finally:
        unet.train()
    seg_img_paso = sum(tiempos) / len(tiempos) / lote
    if ev.get("guia", 1.0) > 1:
        seg_img_paso *= 2
    pasos = ev.get("pasos_muestreo") or r["pasos_entrenamiento"]
    if cfg["modo"] == "multicondicional":
        img_grilla = 2 * min(ev.get("filas_grilla", 12), len(ds.conteo_combinaciones()))
    elif cfg["modo"] == "condicional":
        img_grilla = 2 * len(ds.clases)
    else:
        img_grilla = 8
    n_grillas = math.ceil(e["epocas"] / e.get("cada_muestras_epocas", 10))
    min_grilla = img_grilla * pasos * seg_img_paso / 60
    texto = (f"muestreo {ev.get('muestreador', 'ddpm')} {pasos} pasos: ~{min_grilla:.1f} min por grilla "
             f"x {n_grillas} grillas = ~{n_grillas * min_grilla / 60:.1f} h")
    k_final = int((ev.get("muestras_finales") or {}).get("por_combinacion", 4))
    if k_final > 0:
        grupos = (len(ds.conteo_combinaciones()) if cfg["modo"] == "multicondicional" else len(ds.clases))
        maximo = (ev.get("muestras_finales") or {}).get("max_combinaciones")
        grupos = min(grupos, maximo) if maximo else grupos
        texto += (f"; muestras finales ({k_final * grupos} imagenes) "
                  f"~{k_final * grupos * pasos * seg_img_paso / 60:.0f} min")
    if ev.get("calcular_al_final", True):
        n_img = ev.get("muestras_por_clase", 500) * len(ds.clases)
        horas_met = n_img * pasos * seg_img_paso / 3600
        texto += f"; metricas finales ({n_img} imagenes) ~{horas_met:.1f} h"
    else:
        horas_met = 0
    informar("AVISO" if horas_met > 4 else "OK", texto)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("configs", nargs="*", help="archivos .yaml o carpetas de experimentos")
    parser.add_argument("--sin-prueba-gpu", action="store_true", help="no correr el paso de prueba")
    args = parser.parse_args()

    if verificar_paquetes() and args.configs:
        from ddpm import configuracion
        from lanzar import recolectar_configs
        for archivo in recolectar_configs(args.configs):
            try:
                cfg = configuracion.cargar(archivo)
            except Exception as ex:
                titulo(str(archivo))
                informar("ERROR", f"no se pudo leer: {ex}")
                continue
            # en modo por_especie basta con verificar una vez (mismos datos y modelo)
            if cfg.get("modo") == "por_especie" and not configuracion.validar(cfg):
                cfg = dict(cfg, modo="incondicional")
            verificar_experimento(cfg, prueba_gpu=not args.sin_prueba_gpu)

    titulo("Resultado")
    resumen = f"{CONTEO['OK']} OK, {CONTEO['AVISO']} avisos, {CONTEO['ERROR']} errores"
    print(resumen)
    LINEAS.append(resumen)
    (RAIZ / "verificacion_entorno.txt").write_text("\n".join(LINEAS) + "\n", encoding="utf-8")
    sys.exit(1 if CONTEO["ERROR"] else 0)


if __name__ == "__main__":
    main()

"""Estado de cada corrida, informacion del entorno y escritura segura de JSON."""

import datetime as dt
import json
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path

ARCHIVO_ESTADO = "estado.json"


def ahora():
    return dt.datetime.now().isoformat(timespec="seconds")


def escribir_json(ruta, datos):
    ruta = Path(ruta)
    tmp = ruta.with_suffix(ruta.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(datos, f, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp, ruta)


def leer_json(ruta, defecto=None):
    try:
        with open(ruta, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return defecto


def actualizar_estado(carpeta_run, **campos):
    ruta = Path(carpeta_run) / ARCHIVO_ESTADO
    estado = leer_json(ruta, {}) or {}
    estado.update(campos)
    estado["actualizado"] = ahora()
    escribir_json(ruta, estado)
    return estado


def leer_estado(carpeta_run):
    return leer_json(Path(carpeta_run) / ARCHIVO_ESTADO, {}) or {}


def _version(paquete):
    try:
        from importlib.metadata import version
        return version(paquete)
    except Exception:
        return None


def info_entorno():
    info = {
        "fecha": ahora(),
        "host": socket.gethostname(),
        "python": sys.version.split()[0],
        "ejecutable_python": sys.executable,
        "plataforma": platform.platform(),
        "paquetes": {p: _version(p) for p in [
            "torch", "torchvision", "diffusers", "accelerate", "torchmetrics",
            "torch-fidelity", "numpy", "pillow", "pyyaml", "tensorboard",
        ]},
        "variables": {k: os.environ.get(k) for k in ["CUDA_VISIBLE_DEVICES", "CONDA_DEFAULT_ENV", "VIRTUAL_ENV"]},
    }
    try:
        import torch
        info["cuda_disponible"] = torch.cuda.is_available()
        info["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpus"] = [
                {"nombre": torch.cuda.get_device_name(i),
                 "memoria_gb": round(torch.cuda.get_device_properties(i).total_memory / 1e9, 1)}
                for i in range(torch.cuda.device_count())
            ]
            info["bf16_soportado"] = torch.cuda.is_bf16_supported()
    except Exception as e:
        info["error_torch"] = repr(e)
    try:
        raiz = Path(__file__).resolve().parent.parent
        info["git_commit"] = subprocess.run(
            ["git", "-C", str(raiz), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip() or None
    except Exception:
        info["git_commit"] = None
    return info

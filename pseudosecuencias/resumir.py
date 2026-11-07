"""
Junta el estado y las metricas de todas las corridas en runs/resumen.csv
(una fila por corrida y clase). Se ejecuta solo al final de lanzar.py,
pero tambien se puede correr a mano:

    python resumir.py [carpeta_runs]
"""

import csv
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))

from ddpm import configuracion, registro  # noqa: E402

COLUMNAS_METRICAS = ["clase", "n_reales", "n_generadas", "FID", "IS", "IS_std",
                     "SSIM_medio_gen_real", "SSIM_vecino_gen_real",
                     "SSIM_medio_real_real", "SSIM_vecino_real_real", "pct_gen_SSIM_mayor_0.9"]


def _fila_base(run_dir, cfg, estado):
    r, e = cfg.get("ruido", {}), cfg.get("entrenamiento", {})
    diag = estado.get("diagnostico_ruido") or {}
    return {
        "corrida": run_dir.name,
        "experimento": cfg.get("experimento_padre", cfg.get("nombre")),
        "estado": estado.get("estado", "desconocido"),
        "modo": cfg.get("modo"),
        "condiciones": "+".join(cfg.get("condiciones") or []) or None,
        "resolucion": f"{cfg['datos']['alto']}x{cfg['datos']['ancho']}",
        "pasos": r.get("pasos_entrenamiento"),
        "esquema": r.get("esquema"),
        "snr_terminal_cero": r.get("snr_terminal_cero"),
        "prediccion": r.get("prediccion"),
        "escala_entrada": r.get("escala_entrada"),
        "pct_senal_T": round(diag["porcentaje_senal_T"], 5) if "porcentaje_senal_T" in diag else None,
        "lr": e.get("lr"),
        "lote_efectivo": estado.get("lote_efectivo"),
        "epocas": e.get("epocas"),
        "ema": (e.get("ema") or {}).get("usar"),
        "precision": e.get("precision"),
        "pasos_hechos": estado.get("paso_global"),
        "duracion_horas": estado.get("duracion_horas"),
    }


def resumir(carpeta_runs):
    carpeta_runs = Path(carpeta_runs)
    filas = []
    for run_dir in sorted(p for p in carpeta_runs.iterdir() if p.is_dir() and not p.name.startswith("_")) \
            if carpeta_runs.is_dir() else []:
        if not (run_dir / "config.yaml").exists():
            continue
        cfg = configuracion.cargar(run_dir / "config.yaml")
        base = _fila_base(run_dir, cfg, registro.leer_estado(run_dir))
        metricas = registro.leer_json(run_dir / "metricas.json") or []
        if metricas:
            for m in metricas:
                filas.append({**base, **{k: m.get(k) for k in COLUMNAS_METRICAS}})
        else:
            filas.append({**base, **{k: None for k in COLUMNAS_METRICAS}})
    if not filas:
        return None
    ruta = carpeta_runs / "resumen.csv"
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)
    return ruta


if __name__ == "__main__":
    ruta = resumir(sys.argv[1] if len(sys.argv) > 1 else RAIZ / "runs")
    print(ruta or "No hay corridas que resumir.")

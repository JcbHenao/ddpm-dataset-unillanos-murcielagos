"""
Puente con el pipeline de limpieza: importa SU config.py (y su modulo comun)
en vez de copiar rutas. Asi, si una ruta cambia en el pipeline, cambia aqui
tambien, y nunca quedan desincronizadas.

Orden para encontrar la carpeta del pipeline:
  1. variable de entorno PIPELINE_LIMPIEZA_DIR
  2. pipeline_dir en parametros.yaml

La variable ANOTACIONES_BASE del pipeline sigue funcionando igual
(para correr con otra ruta base sin tocar config.py).
"""

import os
import sys
from pathlib import Path

import yaml

RAIZ = Path(__file__).resolve().parent


def cargar_parametros(ruta=None):
    with open(ruta or RAIZ / "parametros.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


PARAMETROS = cargar_parametros()

PIPELINE_DIR = Path(os.environ.get("PIPELINE_LIMPIEZA_DIR") or PARAMETROS["pipeline_dir"]).resolve()

if not (PIPELINE_DIR / "config.py").exists():
    raise SystemExit(
        f"No se encontro config.py del pipeline en {PIPELINE_DIR}.\n"
        "Ajustar pipeline_dir en parametros.yaml o exportar PIPELINE_LIMPIEZA_DIR."
    )

sys.path.insert(0, str(PIPELINE_DIR))
try:
    import config as config_pipeline                     # noqa: E402
    import espectrogramas_secuencias_comun as comun      # noqa: E402
except Exception as e:  # dependencias del pipeline que falten en este entorno
    raise SystemExit(
        f"No se pudo importar config.py del pipeline ({PIPELINE_DIR}): {type(e).__name__}: {e}\n"
        "Revisar que este entorno tenga instalados requirements.txt."
    )
finally:
    sys.path.remove(str(PIPELINE_DIR))

# ---------------- rutas de entrada (solo lectura) ----------------
RUTA_BASE = Path(config_pipeline.RUTA_BASE)
ARCHIVO_MAESTRO = Path(config_pipeline.ARCHIVO_MAESTRO)
RUTA_AUDIOS = Path(config_pipeline.RUTA_AUDIOS)
RUTA_GENERACION = Path(config_pipeline.RUTA_GENERACION_ESPECTROGRAMAS)

# ---------------- utilidades reutilizadas del pipeline ----------------
limpiar_nombre = comun.limpiar_nombre
indexar_audios = comun.indexar_audios

# ---------------- salidas de este proyecto ----------------
# Por defecto todo queda dentro de la carpeta del proyecto: <proyecto>/datos/
SALIDA = Path(PARAMETROS.get("salida") or Path(__file__).resolve().parent / "datos")
SALIDA_P0 = SALIDA / "P0_verificacion"
SALIDA_P1 = SALIDA / "P1_vinculacion"
SALIDA_P2 = SALIDA / "P2_intervalos"
SALIDA_P3 = SALIDA / "P3_grupos"
SALIDA_P4 = SALIDA / "P4_dataset"
SALIDA_C1 = SALIDA / "C1_revision"

ARCHIVO_PULSOS_VINCULADOS = SALIDA_P1 / "pulsos_vinculados.csv"
ARCHIVO_PULSOS_FRECUENCIA = SALIDA_P2 / "pulsos_con_frecuencia.csv"
ARCHIVO_INTERVALOS = SALIDA_P2 / "intervalos_por_especie.csv"
ARCHIVO_GRUPOS = SALIDA_P3 / "grupos.csv"
ARCHIVO_VENTANA = SALIDA_P3 / "ventana.json"
ARCHIVO_METADATOS = SALIDA_P4 / "metadatos.csv"


def resumen():
    return {
        "pipeline_dir": PIPELINE_DIR,
        "RUTA_BASE": RUTA_BASE,
        "ARCHIVO_MAESTRO": ARCHIVO_MAESTRO,
        "RUTA_AUDIOS": RUTA_AUDIOS,
        "RUTA_GENERACION_ESPECTROGRAMAS": RUTA_GENERACION,
        "SALIDA (este proyecto)": SALIDA,
    }

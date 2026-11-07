"""Utilidades compartidas por las etapas: lectura del maestro y reportes."""

import datetime as dt
import sys
from pathlib import Path

import pandas as pd

import rutas


class Reporte:
    """Imprime y guarda a la vez (cada etapa deja su .txt junto a sus salidas)."""

    def __init__(self, ruta):
        self.ruta = Path(ruta)
        self.ruta.parent.mkdir(parents=True, exist_ok=True)
        self.lineas = [f"# {self.ruta.stem}  ({dt.datetime.now():%Y-%m-%d %H:%M})"]
        self.errores = 0
        self.avisos = 0

    def __call__(self, texto="", nivel=None):
        if nivel == "ERROR":
            self.errores += 1
        elif nivel == "AVISO":
            self.avisos += 1
        linea = f"[{nivel:<5}] {texto}" if nivel else texto
        print(linea, flush=True)
        self.lineas.append(linea)

    def titulo(self, texto):
        self("")
        self(f"=== {texto} ===")

    def guardar(self):
        self.ruta.write_text("\n".join(self.lineas) + "\n", encoding="utf-8")
        return self.ruta


def cargar_maestro(columnas_obligatorias=True):
    """
    Lee el archivo maestro y lo devuelve con nombres de columna internos:
    tipo, archivo, inicio, fin, especie, calidad, tipo_pulso, id
    (+ freq_baja, freq_alta, freq_pico_raven y anotador si existen).
    La especie se normaliza con limpiar_nombre del pipeline, igual que los nombres de carpeta.
    """
    p = rutas.PARAMETROS
    col = p["columnas"]
    df = pd.read_csv(rutas.ARCHIVO_MAESTRO)
    opcionales = ("freq_baja", "freq_alta", "freq_pico_raven", "anotador")
    faltan = [v for k, v in col.items() if k not in opcionales and v not in df.columns]
    if faltan and columnas_obligatorias:
        sys.exit(f"Faltan columnas en {rutas.ARCHIVO_MAESTRO}: {faltan}\n"
                 f"Columnas disponibles: {list(df.columns)}\nAjustar 'columnas' en parametros.yaml.")
    renombrar = {v: k for k, v in col.items() if v in df.columns}
    df = df.rename(columns=renombrar)
    df["especie"] = df["especie"].astype(str).map(rutas.limpiar_nombre)
    for c in ("inicio", "fin", "freq_baja", "freq_alta", "freq_pico_raven"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "id" not in df.columns:
        df["id"] = range(len(df))
    return df


def separar(df):
    vt = rutas.PARAMETROS["valores_tipo"]
    return df[df["tipo"] == vt["pulso"]].copy(), df[df["tipo"] == vt["secuencia"]].copy()

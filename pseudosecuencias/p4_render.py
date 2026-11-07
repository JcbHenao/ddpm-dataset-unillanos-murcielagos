"""
P4 - Renderizar las pseudo-secuencias con la ventana unica de P3.

    python p4_render.py               # falla si P4_dataset ya tiene imagenes
    python p4_render.py --sobrescribir

Cada imagen es audio real desde margen_ms antes del primer pulso hasta
margen_ms despues del ultimo (con el silencio real entre pulsos), alineado a
la izquierda; las columnas que sobran quedan en el piso. Todas miden
exactamente alto x ancho de ventana.json.

Salida en <SALIDA>/P4_dataset/:
  <particion>/<especie>/<audio>__t<inicio>s__n<n>__g<id>.png
  metadatos.csv   una fila por imagen (grupo, tiempos, columnas reales, posicion de cada pulso)
  render.txt      reporte
La carpeta <SALIDA>/P4_dataset/train es la que se pone como datos.ruta en entrenamiento_ddpm.
"""

import argparse
import json
import multiprocessing as mp
import os
import shutil
from collections import Counter

import numpy as np
import pandas as pd
import soundfile as sf
from PIL import Image

import render
import rutas
from comun import Reporte

P = rutas.PARAMETROS


def renderizar_audio(args):
    ruta_audio, grupos, ventana, salida = args
    try:
        audio, sr = sf.read(ruta_audio, dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
    except Exception as e:
        return [], [f"{ruta_audio}: {e}"]
    geo = {k: ventana[k] for k in ("sr", "n_fft", "bin_bajo", "bin_alto", "alto_px")}
    m = ventana["margen_ms"] / 1000
    filas, errores = [], []
    for g in grupos:
        try:
            t0, t1 = g["inicio_s"] - m, g["fin_s"] + m
            x = render.extraer(audio, sr, t0, t1, ventana["sr"])
            db, columnas = render.espectrograma_db(x, geo, ventana["hop"], ventana["ancho_px"], ventana["rango_db"])
            img = render.a_imagen(db, ventana["rango_db"], ventana["colormap"])
            base = rutas.limpiar_nombre(os.path.splitext(g["archivo"])[0])
            nombre = f"{base}__t{g['inicio_s']:.3f}s__n{g['n']}__g{g['grupo']:06d}.png"
            carpeta = os.path.join(salida, g["particion"], g["especie"])
            os.makedirs(carpeta, exist_ok=True)
            img.save(os.path.join(carpeta, nombre))
            # columna donde empieza cada pulso (para revisar y para calibrar el detector)
            cols = [round((float(s) - t0) * 1000 / ventana["ms_por_px"], 1) for s in g["inicios_s"].split(";")]
            filas.append({**g, "archivo_imagen": nombre, "ruta_relativa": f"{g['particion']}/{g['especie']}/{nombre}",
                          "sample_rate_original": sr, "inicio_ventana_s": round(t0, 6),
                          "columnas_con_audio": columnas, "columnas_inicio_pulsos": ";".join(map(str, cols)),
                          "nyquist_bajo_banda": sr / 2 < ventana["banda_hz"][1]})
        except Exception as e:
            errores.append(f"{g['archivo']} grupo {g['grupo']}: {type(e).__name__}: {e}")
    return filas, errores


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sobrescribir", action="store_true", help="borrar P4_dataset antes de renderizar")
    args = parser.parse_args()

    for req in (rutas.ARCHIVO_GRUPOS, rutas.ARCHIVO_VENTANA):
        if not req.exists():
            raise SystemExit(f"Falta {req}: correr antes p3_grupos.py")
    if rutas.SALIDA_P4.exists() and any(rutas.SALIDA_P4.iterdir()):
        if not args.sobrescribir:
            raise SystemExit(f"{rutas.SALIDA_P4} ya tiene contenido. Usar --sobrescribir para rehacerlo "
                             "(solo se borra esa carpeta, que es salida de este proyecto).")
        shutil.rmtree(rutas.SALIDA_P4)
    rutas.SALIDA_P4.mkdir(parents=True, exist_ok=True)
    rep = Reporte(rutas.SALIDA_P4 / "render.txt")

    with open(rutas.ARCHIVO_VENTANA, encoding="utf-8") as f:
        ventana = json.load(f)
    grupos = pd.read_csv(rutas.ARCHIVO_GRUPOS)
    grupos = grupos[grupos["incluido"]].reset_index(drop=True)
    grupos["grupo"] = grupos.index
    indice = rutas.indexar_audios(str(rutas.RUTA_AUDIOS))

    tareas = [(indice[a], g.to_dict("records"), ventana, str(rutas.SALIDA_P4))
              for a, g in grupos.groupby("archivo")]
    procesos = P.get("procesos") or os.cpu_count()
    rep(f"Renderizando {len(grupos)} grupos de {len(tareas)} audios con {procesos} procesos; "
        f"{ventana['alto_px']} x {ventana['ancho_px']} px, {ventana['duracion_ms']:.1f} ms por imagen")
    rep(f"Intensidad como el pipeline previo: dB respecto al maximo del espectro completo, "
        f"recorte [-{ventana['rango_db']:g}, 0] dB, colormap {ventana['colormap']}, sin filtros")

    filas, errores = [], []
    with mp.Pool(processes=procesos) as pool:
        for f, e in pool.imap_unordered(renderizar_audio, tareas):
            filas.extend(f)
            errores.extend(e)
    for e in errores:
        rep(e, "ERROR")

    meta = pd.DataFrame(filas).sort_values(["particion", "especie", "archivo", "inicio_s"])
    meta.to_csv(rutas.ARCHIVO_METADATOS, index=False)

    tamanos = Counter()
    for r in meta["ruta_relativa"]:
        with Image.open(rutas.SALIDA_P4 / r) as im:
            tamanos[(im.height, im.width)] += 1
    esperado = (ventana["alto_px"], ventana["ancho_px"])
    if set(tamanos) == {esperado}:
        rep(f"las {len(meta)} imagenes miden {esperado[0]} x {esperado[1]} px (alto x ancho)", "OK")
    else:
        rep(f"tamanos distintos al esperado {esperado}: {dict(tamanos)}", "ERROR")
    rep(f"{int(meta['nyquist_bajo_banda'].sum())} imagenes de audios con Nyquist bajo la banda (parte alta en el piso)",
        "AVISO" if meta["nyquist_bajo_banda"].any() else "OK")

    rep.titulo("Imagenes por particion, especie y n")
    rep(meta.groupby(["particion", "especie", "n"]).size().unstack(fill_value=0).to_string())
    rep(f"\nDataset: {rutas.SALIDA_P4}  (para entrenar: datos.ruta = {rutas.SALIDA_P4 / 'train'})")
    rep.guardar()


if __name__ == "__main__":
    main()

"""
Control 1 - Revisar el dataset ANTES de entrenar.

    python c1_revisar.py

Comprobaciones automaticas:
  - todas las imagenes existen y miden exactamente alto x ancho de ventana.json
  - ninguna grabacion aparece en dos particiones
  - todas las especies tienen grabaciones en train, val y test
  - los intervalos del dataset representan los de todos los pares validos (P2)
Para la revision visual:
  - hojas/<especie>.png: muestras de 2 y 3 pulsos de train, con marcas en el
    inicio anotado de cada pulso (las marcas solo estan en la hoja, no en el dataset)

Salidas en <SALIDA>/C1_revision/. El reporte termina con la lista a aprobar.
"""

import json

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

import rutas
from comun import Reporte

POR_N = 6   # muestras por especie y n en cada hoja


def hoja(meta_esp, ventana, ruta, semilla=0):
    filas = []
    for n, g in meta_esp.groupby("n"):
        filas += g.sample(min(POR_N, len(g)), random_state=semilla).to_dict("records")
    if not filas:
        return
    alto, ancho, banda = ventana["alto_px"], ventana["ancho_px"], 18
    lienzo = Image.new("RGB", (ancho, len(filas) * (alto + banda)), "white")
    dib = ImageDraw.Draw(lienzo)
    for k, f in enumerate(filas):
        y = k * (alto + banda)
        dib.text((4, y + 3), f"n={f['n']}  {f.get('patron', '')}  {f['archivo']}  t={f['inicio_s']:.3f}s  intervalos {f['intervalos_ms']} ms  "
                             f"kHz {f['freqs_khz']}", fill="black")
        with Image.open(rutas.SALIDA_P4 / f["ruta_relativa"]) as im:
            lienzo.paste(im.convert("RGB"), (0, y + banda))
        for c in str(f["columnas_inicio_pulsos"]).split(";"):
            x = float(c)
            dib.polygon([(x - 4, y + banda), (x + 4, y + banda), (x, y + banda + 6)], fill="white")
        fin_audio = int(f["columnas_con_audio"])
        dib.line([(fin_audio, y + banda), (fin_audio, y + banda + alto)], fill="white", width=1)
    lienzo.save(ruta)


def main():
    rep = Reporte(rutas.SALIDA_C1 / "revision.txt")
    if not rutas.ARCHIVO_METADATOS.exists():
        raise SystemExit(f"Falta {rutas.ARCHIVO_METADATOS}: correr antes p4_render.py")
    with open(rutas.ARCHIVO_VENTANA, encoding="utf-8") as f:
        ventana = json.load(f)
    meta = pd.read_csv(rutas.ARCHIVO_METADATOS)
    pulsos = pd.read_csv(rutas.ARCHIVO_PULSOS_FRECUENCIA)
    checks = {}

    rep.titulo("Tamano de las imagenes")
    esperado, malos, faltan = (ventana["alto_px"], ventana["ancho_px"]), 0, 0
    for r in meta["ruta_relativa"]:
        p = rutas.SALIDA_P4 / r
        if not p.exists():
            faltan += 1
            continue
        with Image.open(p) as im:
            malos += (im.height, im.width) != esperado
    checks["tamano_unico"] = malos == 0 and faltan == 0
    rep(f"{len(meta)} imagenes; {faltan} faltantes; {malos} con tamano distinto de {esperado[0]} x {esperado[1]}",
        "OK" if checks["tamano_unico"] else "ERROR")

    rep.titulo("Particion por grabacion")
    por_audio = meta.groupby("archivo")["particion"].nunique()
    fuga = por_audio[por_audio > 1]
    checks["sin_fuga"] = fuga.empty
    rep(f"{len(fuga)} grabaciones en mas de una particion", "OK" if fuga.empty else "ERROR")
    cobertura = meta.groupby(["especie", "particion"])["archivo"].nunique().unstack(fill_value=0)
    for s in ("train", "val", "test"):
        if s not in cobertura.columns:
            cobertura[s] = 0
    cobertura = cobertura[["train", "val", "test"]]
    rep("grabaciones por especie y particion:\n" + cobertura.to_string())
    incompletas = cobertura[(cobertura == 0).any(axis=1)].index.tolist()
    checks["todas_las_particiones"] = not incompletas
    if incompletas:
        rep(f"especies sin grabaciones en alguna particion: {incompletas}", "AVISO")
    rep("imagenes por especie, n y particion:\n" +
        meta.groupby(["especie", "n", "particion"]).size().unstack(fill_value=0).to_string())

    rep.titulo("Patrones de frecuencia (condicion del modelo)")
    if "patron" in meta.columns:
        tabla_p = meta[meta["particion"] == "train"].groupby(["especie", "n", "patron"]).size().unstack(fill_value=0)
        rep("imagenes de train por especie, n y patron:\n" + tabla_p.to_string())
        escasos = [(e, n, p) for (e, n), fila in tabla_p.iterrows() for p, c in fila.items() if 0 < c < 20]
        if escasos:
            rep(f"{len(escasos)} combinaciones especie/n/patron con menos de 20 imagenes en train: el modelo "
                f"las aprendera sobre todo por lo que comparten con otras (embeddings separados por condicion)", "AVISO")
    else:
        rep("metadatos sin columna patron: volver a correr p3_grupos.py y p4_render.py", "ERROR")

    rep.titulo("Representatividad de los intervalos (ms)")
    validos = pulsos[pulsos["intervalo_valido"]]
    filas = []
    for esp, g in meta.groupby("especie"):
        dataset = np.array([float(v) for s in g["intervalos_ms"] for v in str(s).split(";")])
        todos = validos[validos["especie"] == esp]["intervalo_anterior_s"].to_numpy() * 1000
        if not len(todos):
            continue
        filas.append({"especie": esp,
                      "dataset_p5": round(np.percentile(dataset, 5), 1), "todos_p5": round(np.percentile(todos, 5), 1),
                      "dataset_p50": round(np.median(dataset), 1), "todos_p50": round(np.median(todos), 1),
                      "dataset_p95": round(np.percentile(dataset, 95), 1), "todos_p95": round(np.percentile(todos, 95), 1),
                      "pct_sube": round(100 * np.mean([t == "sube" for s in g["transiciones"] for t in str(s).split(";")]), 1)})
    tabla = pd.DataFrame(filas)
    rep(tabla.to_string(index=False))
    sesgo = tabla[(tabla["dataset_p50"] - tabla["todos_p50"]).abs() > 0.2 * tabla["todos_p50"]]
    checks["intervalos_representativos"] = sesgo.empty
    if not sesgo.empty:
        rep(f"la mediana del dataset se aleja mas de 20 % de la de todos los pares en: {sesgo['especie'].tolist()} "
            "(revisar si el filtro de calidad o de ventana sesga los intervalos)", "AVISO")

    rep.titulo("Hojas de contacto")
    carpeta = rutas.SALIDA_C1 / "hojas"
    carpeta.mkdir(parents=True, exist_ok=True)
    for esp, g in meta[meta["particion"] == "train"].groupby("especie"):
        hoja(g, ventana, carpeta / f"{esp}.png")
    rep(f"en {carpeta}: triangulos blancos = inicio anotado de cada pulso; linea blanca = fin del audio real")

    rep.titulo("Lista para aprobar el Control 1")
    rep(f"[{'x' if checks['tamano_unico'] else ' '}] todas las imagenes miden {esperado[0]} x {esperado[1]} px")
    rep(f"[{'x' if checks['sin_fuga'] else ' '}] ninguna grabacion en dos particiones")
    rep(f"[{'x' if checks['todas_las_particiones'] else ' '}] todas las especies en train, val y test")
    rep(f"[{'x' if checks['intervalos_representativos'] else ' '}] intervalos del dataset representativos")
    rep("[ ] revision visual: cada triangulo cae sobre un pulso y el silencio entre pulsos se ve")
    rep("[ ] revision visual: la alternancia de frecuencia de cada especie se distingue")
    rep.guardar()


if __name__ == "__main__":
    main()

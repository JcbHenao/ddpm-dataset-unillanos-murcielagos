"""
Secuencias de prueba para mostrar: al terminar una corrida se generan imagenes
para cada combinacion de condiciones (especie, n, patron) y se ponen al lado de
imagenes reales de la misma combinacion.

Salida en runs/<corrida>/muestras_finales/:
  <especie>/<combinacion>/generada_XX.png   imagenes generadas (PNG del mismo tamano que el dataset)
  <especie>/<combinacion>/real_XX.png       copias de imagenes reales de train con esa combinacion
  hoja_<especie>.png                        por combinacion: fila de reales y fila de generadas
  indice.csv                                una fila por imagen (tipo, condiciones, archivo, origen)

Se controla con evaluacion.muestras_finales en el YAML:
  por_combinacion: imagenes generadas (y reales) por combinacion; 0 = no generar
  max_combinaciones: null = todas las del dataset; N = las N mas frecuentes
"""

import csv
import random
import shutil
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from .muestreo import a_pil, generar

ALTO_TEXTO = 16
SEPARACION = 6


def _grupos(cfg, dataset, max_combinaciones):
    """Lista de (nombre_especie, nombre_combinacion, descripcion, etiqueta o None, indices_reales)."""
    grupos = []
    if cfg["modo"] == "multicondicional":
        conteo = dataset.conteo_combinaciones()
        combos = [c for c, _ in conteo.most_common(max_combinaciones)]
        combos.sort(key=lambda c: (str(c[dataset.condiciones.index("especie")]) if "especie" in dataset.condiciones
                                   else "", -conteo[c]))
        valores_df = dataset.df[dataset.condiciones].astype(str).to_numpy()
        for combo in combos:
            combo_s = tuple(str(v) for v in combo)
            indices = [i for i, fila in enumerate(valores_df) if tuple(fila) == combo_s]
            etiqueta = torch.tensor([dataset.vocab[c].index(v) for c, v in zip(dataset.condiciones, combo_s)],
                                    dtype=torch.long)
            valores = dict(zip(dataset.condiciones, combo_s))
            especie = valores.get("especie", "todas")
            nombre = "_".join(f"n{v}" if c == "n" else v for c, v in valores.items() if c != "especie") or "todas"
            desc = ", ".join(f"{c}={v}" for c, v in valores.items()) + f"  ({conteo[combo]} reales en train)"
            grupos.append((especie, nombre, desc, etiqueta, indices))
    elif cfg["modo"] == "condicional":
        for i, clase in enumerate(dataset.clases):
            indices = dataset.indices_de_clase(i)
            grupos.append((clase, "todas", f"especie={clase}  ({len(indices)} reales)",
                           torch.tensor(i, dtype=torch.long), indices))
    else:
        indices = list(range(len(dataset)))
        grupos.append((dataset.clases[0], "todas", f"{dataset.clases[0]}  ({len(indices)} reales)", None, indices))
    return grupos


def _hoja(bloques, ancho, alto, columnas):
    """bloques: lista de (descripcion, [PIL reales], [PIL generadas]) -> una imagen."""
    alto_bloque = ALTO_TEXTO + 2 * (ALTO_TEXTO + alto) + SEPARACION
    hoja = Image.new("RGB", (columnas * ancho, alto_bloque * len(bloques)), "white")
    dib = ImageDraw.Draw(hoja)
    for b, (desc, reales, generadas) in enumerate(bloques):
        y = b * alto_bloque
        dib.text((4, y + 2), desc, fill="black")
        y += ALTO_TEXTO
        for titulo, imgs in (("reales (train)", reales), ("generadas", generadas)):
            dib.text((4, y + 2), titulo if imgs else f"{titulo}: no hay", fill=(90, 90, 90))
            y += ALTO_TEXTO
            for k, im in enumerate(imgs[:columnas]):
                hoja.paste(im.convert("RGB"), (k * ancho, y))
            y += alto
    return hoja


def generar_muestras_finales(cfg, unet, scheduler_muestreo, dataset, carpeta, imprimir=print):
    ev = cfg.get("evaluacion", {})
    opciones = ev.get("muestras_finales", {}) or {}
    k = int(opciones.get("por_combinacion", 4))
    if k <= 0:
        return None
    d, r = cfg["datos"], cfg["ruido"]
    carpeta = Path(carpeta)
    if carpeta.exists():
        shutil.rmtree(carpeta)
    carpeta.mkdir(parents=True)

    grupos = _grupos(cfg, dataset, opciones.get("max_combinaciones"))
    # todas las etiquetas en un solo lote continuo (mas rapido que combinacion por combinacion)
    etiquetas = None
    if grupos[0][3] is not None:
        etiquetas = torch.stack([g[3] for g in grupos for _ in range(k)])
    n_total = k * len(grupos)
    pasos = ev.get("pasos_muestreo") or r["pasos_entrenamiento"]
    imprimir(f"[{cfg['nombre']}] muestras finales: {len(grupos)} combinaciones x {k} = {n_total} imagenes, "
             f"{ev.get('muestreador', 'ddpm')} {pasos} pasos")
    generator = torch.Generator().manual_seed(cfg["entrenamiento"]["semilla"] + 7)
    lote = ev.get("lote", 16)
    generadas = []
    for i in range(0, n_total, lote):
        b = min(lote, n_total - i)
        imgs = generar(unet, scheduler_muestreo, b, d["alto"], d["ancho"], d.get("canales", 3),
                       r.get("escala_entrada", 1.0), etiquetas[i:i + b] if etiquetas is not None else None,
                       ev.get("pasos_muestreo"), generator, guia=ev.get("guia", 1.0),
                       precision=cfg["entrenamiento"].get("precision"),
                       descripcion=f"[{cfg['nombre']}] muestras finales {i + b}/{n_total}")
        generadas.extend(a_pil(imgs))

    azar = random.Random(cfg["entrenamiento"]["semilla"])
    filas_indice, hojas = [], {}
    for j, (especie, nombre, desc, etiqueta, indices) in enumerate(grupos):
        destino = carpeta / especie / nombre
        destino.mkdir(parents=True, exist_ok=True)
        propias = generadas[j * k:(j + 1) * k]
        for m, im in enumerate(propias):
            im.save(destino / f"generada_{m:02d}.png")
            filas_indice.append({"tipo": "generada", "especie": especie, "combinacion": nombre,
                                 "archivo": f"{especie}/{nombre}/generada_{m:02d}.png", "origen": ""})
        reales = []
        for m, idx in enumerate(azar.sample(indices, min(k, len(indices)))):
            origen = Path(dataset.archivos[idx][0])
            shutil.copy2(origen, destino / f"real_{m:02d}.png")
            reales.append(Image.open(destino / f"real_{m:02d}.png").copy())
            filas_indice.append({"tipo": "real", "especie": especie, "combinacion": nombre,
                                 "archivo": f"{especie}/{nombre}/real_{m:02d}.png", "origen": str(origen)})
        hojas.setdefault(especie, []).append((desc, reales, propias))

    for especie, bloques in hojas.items():
        _hoja(bloques, d["ancho"], d["alto"], k).save(carpeta / f"hoja_{especie}.png")
    with open(carpeta / "indice.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["tipo", "especie", "combinacion", "archivo", "origen"])
        w.writeheader()
        w.writerows(filas_indice)
    imprimir(f"[{cfg['nombre']}] muestras finales en {carpeta} ({len(hojas)} hojas, una por especie)")
    return carpeta


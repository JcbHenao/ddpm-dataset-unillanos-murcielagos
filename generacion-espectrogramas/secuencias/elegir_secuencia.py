#!/usr/bin/env python3
"""Elige automaticamente buenas secuencias de pulsos para las diapositivas.

Lee las tablas de anotaciones (Raven, .txt separado por tabuladores), busca el
audio de cada fila, encuentra en cada secuencia el mejor trozo de 1 s (varios
pulsos nitidos y sin ruido) y genera, para los mejores:
  - NN_..._secuencia.png : espectrograma con ejes (ms, kHz, dB)  -> diapositiva 4
  - NN_..._pulso.png     : zoom al pulso mas fuerte              -> diapositiva 5
  - hoja_contacto.png    : todas las candidatas en una sola imagen para elegir a ojo
  - ranking.csv          : tabla con puntaje, archivo, inicio, fin, n.o de pulsos

Uso:
  python elegir_secuencia.py \
      --audios "/media/adalab/bodega/CamiloBuitrago/audios/MolossusMolossus" \
      --anotaciones "/media/adalab/bodega/CamiloBuitrago/anotaciones/Anotaciones/Secuencias/Jancel/Molossus molossus" \
      --n 9 -o candidatas

Opciones utiles: --ventana 1.0 (s que se dibujan, el mejor trozo de cada secuencia), --dur-min/--dur-max (duracion de la secuencia anotada),
--max-eval 300 (cuantas secuencias evaluar como maximo), --fmax 80 (kHz mostrados).
Requisitos: numpy, scipy, matplotlib.
"""
import argparse
import csv
import os
import random
import sys

import numpy as np
from scipy.io import wavfile
from scipy.signal import spectrogram, find_peaks
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------- tablas
def _col(header, *claves):
    """Indice de la primera columna cuyo nombre empieza por alguna clave."""
    for i, h in enumerate(header):
        hl = h.strip().lower()
        if any(hl.startswith(c) for c in claves):
            return i
    return None


def leer_tablas(carpeta):
    filas = []
    for raiz, _, nombres in os.walk(carpeta):
        for nom in sorted(nombres):
            if not nom.lower().endswith((".txt", ".csv", ".tsv")):
                continue
            ruta = os.path.join(raiz, nom)
            try:
                with open(ruta, newline="", encoding="utf-8-sig", errors="replace") as fh:
                    muestra = fh.read(4096)
                    fh.seek(0)
                    delim = "\t" if muestra.count("\t") >= muestra.count(",") else ","
                    rd = csv.reader(fh, delimiter=delim)
                    header = next(rd, None)
                    if not header:
                        continue
                    ib = _col(header, "begin time")
                    ie = _col(header, "end time")
                    if ib is None or ie is None:
                        continue
                    iff = _col(header, "begin file", "file")
                    il = _col(header, "low freq")
                    ih = _col(header, "high freq")
                    for r in rd:
                        try:
                            ini, fin = float(r[ib]), float(r[ie])
                        except (ValueError, IndexError):
                            continue
                        arch = r[iff].strip() if iff is not None and iff < len(r) else \
                            os.path.splitext(nom)[0] + ".wav"
                        try:
                            lo = float(r[il]) if il is not None else 20000.0
                            hi = float(r[ih]) if ih is not None else 40000.0
                        except (ValueError, IndexError):
                            lo, hi = 20000.0, 40000.0
                        filas.append(dict(tabla=nom, archivo=os.path.basename(arch.replace("\\", "/")),
                                          ini=ini, fin=fin, lo=lo, hi=hi))
            except OSError as e:
                print("No pude leer", ruta, e, file=sys.stderr)
    return filas


def indexar_audios(carpeta):
    idx = {}
    for raiz, _, nombres in os.walk(carpeta):
        for nom in nombres:
            if nom.lower().endswith(".wav"):
                idx[nom.lower()] = os.path.join(raiz, nom)
    return idx


# ---------------------------------------------------------------- audio
def leer_segmento(ruta, t0, t1):
    try:
        fs, x = wavfile.read(ruta, mmap=True)
    except Exception:
        fs, x = wavfile.read(ruta)
    a, b = max(0, int(t0 * fs)), min(len(x), int(t1 * fs))
    seg = np.array(x[a:b])
    if seg.ndim > 1:
        seg = seg[:, 0]
    if np.issubdtype(seg.dtype, np.integer):
        seg = seg.astype(np.float64) / float(np.iinfo(seg.dtype).max)
    else:
        seg = seg.astype(np.float64)
    return fs, seg


def stft(seg, fs, nfft=512, solape=0.75):
    f, t, S = spectrogram(seg, fs=fs, window="hann", nperseg=nfft,
                          noverlap=int(nfft * solape), mode="magnitude")
    return f, t, S


def puntuar(seg, fs, lo, hi, ventana):
    """Busca dentro de la secuencia la mejor ventana de `ventana` s.
    Devuelve (puntaje, n_pulsos_en_ventana, inicio_ventana_s, t_pico_s), tiempos relativos al inicio de `seg`.
    Puntaje = contraste de los pulsos en la banda anotada + bonus por tener varios pulsos nitidos - saturacion."""
    f, t, S = stft(seg, fs, nfft=512, solape=0.5)
    banda = (f >= lo) & (f <= hi)
    mala = (-1e9, 0, 0.0, 0.0)
    if banda.sum() < 2 or len(t) < 3:
        return mala
    e = 20 * np.log10(np.sqrt((S[banda] ** 2).mean(axis=0)) + 1e-12)
    ruido = np.median(e)
    paso = max(t[1] - t[0], 1e-9)
    picos, _ = find_peaks(e, height=ruido + 10, distance=max(1, int(0.004 / paso)))
    if len(picos) < 3:
        return mala
    tp = t[picos]
    dur = len(seg) / fs
    ventana = min(ventana, dur)
    mejor = mala
    # candidatas: ventanas que empiezan un poco antes de cada pulso
    for s0 in np.unique(np.clip(tp - 0.1 * ventana, 0, max(dur - ventana, 0))):
        m = (tp >= s0) & (tp <= s0 + ventana)
        n = int(m.sum())
        if n < 3:
            continue
        pk = e[picos[m]]
        snr = float(np.percentile(pk, 75) - ruido)
        a0, a1 = int(s0 * fs), int((s0 + ventana) * fs)
        saturado = float((np.abs(seg[a0:a1]) > 0.98).mean())
        score = min(snr, 40.0) + 2.0 * min(n, 8) - 100.0 * saturado
        if score > mejor[0]:
            mejor = (score, n, float(s0), float(tp[m][np.argmax(pk)]))
    return mejor


# ---------------------------------------------------------------- dibujo
def dibujar(ax, seg, fs, fmax, rango_db=70, cmap="jet", size=12, etiquetas=True, vmax=None):
    f, t, S = stft(seg, fs)
    SdB = 20 * np.log10(S + 1e-12)
    vmax = SdB.max() if vmax is None else vmax
    im = ax.imshow(SdB, origin="lower", aspect="auto", cmap=cmap, vmin=vmax - rango_db, vmax=vmax,
                   extent=[t[0] * 1000, t[-1] * 1000, f[0] / 1000, f[-1] / 1000])
    ax.set_ylim(0, fmax)
    if etiquetas:
        ax.set_xlabel("Tiempo (ms)", fontsize=size, labelpad=2)
        ax.set_ylabel("Frecuencia (kHz)", fontsize=size, labelpad=2)
        ax.tick_params(labelsize=size - 1, length=3, pad=2)
    return im


def guardar_figura(seg, fs, fmax, ruta, letra=10, barra=False):
    """Figura ancha para diapositiva: letra moderada y barra de color delgada para no robar espacio."""
    fig, ax = plt.subplots(figsize=(12, 4.4))
    im = dibujar(ax, seg, fs, fmax, size=letra)
    if barra:
        cb = fig.colorbar(im, ax=ax, pad=0.008, fraction=0.025, aspect=30)
        cb.set_label("dB", fontsize=letra)
        cb.ax.tick_params(labelsize=letra - 3)
    fig.subplots_adjust(left=0.055, right=0.982 if not barra else 0.95, bottom=0.13, top=0.985)
    fig.savefig(ruta, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- principal
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audios", required=True)
    ap.add_argument("--anotaciones", required=True)
    ap.add_argument("-o", "--salida", default="candidatas")
    ap.add_argument("--n", type=int, default=9, help="cuantas candidatas dibujar")
    ap.add_argument("--ventana", type=float, default=1.0, help="duracion (s) del trozo que se dibuja (maximo recomendado: 1)")
    ap.add_argument("--dur-min", type=float, default=1.0, help="duracion minima de la secuencia anotada (s)")
    ap.add_argument("--dur-max", type=float, default=12.0, help="duracion maxima de la secuencia anotada (s)")
    ap.add_argument("--max-eval", type=int, default=300)
    ap.add_argument("--margen", type=float, default=0.0, help="segundos extra antes y despues de la ventana")
    ap.add_argument("--fmax", type=float, default=80.0, help="kHz mostrados")
    ap.add_argument("--semilla", type=int, default=42)
    ap.add_argument("--letra", type=int, default=10, help="tamano de letra de los ejes")
    ap.add_argument("--con-barra", action="store_true", help="agrega la barra de color (dB); por defecto no se dibuja")
    ap.add_argument("--sin-barra", action="store_true", help=argparse.SUPPRESS)  # compatibilidad: ya es el valor por defecto
    a = ap.parse_args()

    os.makedirs(a.salida, exist_ok=True)
    print("Leyendo anotaciones...")
    filas = leer_tablas(a.anotaciones)
    print(f"  {len(filas)} filas")
    print("Indexando audios...")
    idx = indexar_audios(a.audios)
    print(f"  {len(idx)} archivos .wav")

    cand = [r for r in filas if a.dur_min <= r["fin"] - r["ini"] <= a.dur_max
            and r["archivo"].lower() in idx]
    print(f"  {len(cand)} secuencias de {a.dur_min}-{a.dur_max} s con audio disponible")
    if not cand:
        sys.exit("No hay candidatas: revisa las rutas, los nombres de archivo o --dur-min/--dur-max.")
    random.Random(a.semilla).shuffle(cand)
    cand = cand[: a.max_eval]

    res = []
    for k, r in enumerate(cand, 1):
        try:
            fs, seg = leer_segmento(idx[r["archivo"].lower()], r["ini"], r["fin"])
            lo = min(r["lo"], fs / 2 * 0.9)
            hi = min(r["hi"], fs / 2 * 0.95)
            sc, npul, v0, tp = puntuar(seg, fs, lo, hi, a.ventana)
        except Exception as e:  # audio corrupto, formato raro...
            print("  omitida", r["archivo"], e, file=sys.stderr)
            continue
        r.update(score=sc, n_pulsos=npul, v0=r["ini"] + v0, v1=r["ini"] + v0 + min(a.ventana, r["fin"] - r["ini"]),
                 t_pico=r["ini"] + tp, fs=fs)
        res.append(r)
        if k % 25 == 0:
            print(f"  evaluadas {k}/{len(cand)}")

    res = [r for r in res if r["score"] > -1e8]
    res.sort(key=lambda r: r["score"], reverse=True)
    top = res[: a.n]

    with open(os.path.join(a.salida, "ranking.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["puesto", "puntaje", "archivo", "ventana_inicio_s", "ventana_fin_s", "pulsos_en_ventana", "secuencia_inicio_s", "secuencia_fin_s", "tabla"])
        for i, r in enumerate(res[:50], 1):
            w.writerow([i, f"{r['score']:.1f}", r["archivo"], f"{r['v0']:.3f}", f"{r['v1']:.3f}", r["n_pulsos"], r["ini"], r["fin"], r["tabla"]])

    cols = 3
    filas_g = int(np.ceil(len(top) / cols))
    fig, axs = plt.subplots(filas_g, cols, figsize=(5 * cols, 3 * filas_g), squeeze=False)
    for ax in axs.ravel():
        ax.axis("off")
    for i, r in enumerate(top, 1):
        ruta = idx[r["archivo"].lower()]
        t0, t1 = max(0.0, r["v0"] - a.margen), r["v1"] + a.margen
        fs, seg = leer_segmento(ruta, t0, t1)
        base = f"{i:02d}_{os.path.splitext(r['archivo'])[0]}_{r['v0']:.2f}-{r['v1']:.2f}"
        guardar_figura(seg, fs, min(a.fmax, fs / 2000), os.path.join(a.salida, base + "_secuencia.png"), a.letra, a.con_barra)
        # zoom al pulso mas fuerte (+-12 ms)
        tc = r["t_pico"]
        fs2, z = leer_segmento(ruta, tc - 0.012, tc + 0.012)
        guardar_figura(z, fs2, min(a.fmax, fs2 / 2000), os.path.join(a.salida, base + "_pulso.png"), a.letra, a.con_barra)
        ax = axs[(i - 1) // cols][(i - 1) % cols]
        ax.axis("on")
        dibujar(ax, seg, fs, min(a.fmax, fs / 2000), etiquetas=False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"#{i}  {r['score']:.0f} pts  {r['n_pulsos']} pulsos\n{r['archivo']}  {r['v0']:.2f}-{r['v1']:.2f} s", fontsize=8)
        print(f"#{i}: {r['archivo']}  ventana {r['v0']:.3f}-{r['v1']:.3f} s  puntaje {r['score']:.1f}  pulsos {r['n_pulsos']}")
    fig.tight_layout()
    fig.savefig(os.path.join(a.salida, "hoja_contacto.png"), dpi=130)
    print("Listo. Mira primero:", os.path.join(a.salida, "hoja_contacto.png"))


if __name__ == "__main__":
    main()

"""
P2 - Frecuencia pico de cada pulso e intervalos entre pulsos por especie.

    python p2_intervalos.py

1. Lee pulsos_vinculados.csv (P1).
2. Abre cada audio una vez y mide la frecuencia pico de cada pulso dentro de la
   banda de trabajo (o dentro de su caja anotada si el maestro trae frecuencias).
3. Clasifica cada par consecutivo por transicion de frecuencia: sube, baja o igual
   (umbral en parametros.yaml). En especies que alternan, la distribucion de
   intervalos es bimodal; separarla por transicion distingue el intervalo dentro
   de un par del intervalo entre pares.
4. Calcula percentiles de intervalo por especie y transicion, solo con pares
   validos: misma especie, sin solapamiento y dentro de la misma secuencia
   (o, sin secuencia anotada, a menos de max_intervalo_sin_secuencia_s).

Salidas en <SALIDA>/P2_intervalos/:
  pulsos_con_frecuencia.csv   P1 + frecuencia pico, transicion y validez del intervalo
  intervalos_por_especie.csv  percentiles (ms) por especie y transicion
  frecuencias_por_especie.csv frecuencia pico y alternancia por especie
  histogramas/<especie>.png
  intervalos.txt              reporte, con el contraste contra rangos publicados
"""

import multiprocessing as mp
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import rutas  # noqa: E402
from comun import Reporte  # noqa: E402

P = rutas.PARAMETROS
F = P["frecuencia"]


def _siguiente_potencia_de_2(n):
    return 1 << (int(n) - 1).bit_length()


def frecuencias_de_un_audio(args):
    """Frecuencia pico (Hz) de cada pulso de un audio. Devuelve lista de (indice, sr, freq, nivel_db)."""
    import librosa

    ruta, filas = args
    try:
        audio, sr = librosa.load(ruta, sr=None, mono=True)
    except Exception as e:
        print(f"Error cargando {ruta}: {e}", flush=True)
        return [(i, np.nan, np.nan, np.nan) for i, *_ in filas]

    nyquist = sr / 2
    salida = []
    for idx, inicio, fin, f_baja, f_alta in filas:
        a, b = int(round(inicio * sr)), int(round(fin * sr))
        seg = audio[max(a, 0):min(b, len(audio))]
        if len(seg) < 8:
            salida.append((idx, sr, np.nan, np.nan))
            continue
        n = max(F["n_fft"], _siguiente_potencia_de_2(len(seg)))
        espectro = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), n=n))
        freqs = np.fft.rfftfreq(n, 1 / sr)
        lo = F["banda_hz"][0] if np.isnan(f_baja) else max(F["banda_hz"][0], f_baja)
        hi = min(F["banda_hz"][1], nyquist) if np.isnan(f_alta) else min(F["banda_hz"][1], f_alta, nyquist)
        mascara = (freqs >= lo) & (freqs <= hi)
        if not mascara.any():
            salida.append((idx, sr, np.nan, np.nan))
            continue
        k = np.flatnonzero(mascara)[np.argmax(espectro[mascara])]
        salida.append((idx, sr, float(freqs[k]), float(20 * np.log10(espectro[k] + 1e-12))))
    return salida


def medir_frecuencias(pulsos, rep):
    indice = rutas.indexar_audios(str(rutas.RUTA_AUDIOS))
    tiene_caja = "freq_baja" in pulsos.columns and "freq_alta" in pulsos.columns
    tareas, faltan = [], 0
    for audio, g in pulsos.groupby("archivo"):
        if audio not in indice:
            faltan += len(g)
            continue
        filas = [(i, r.inicio, r.fin,
                  r.freq_baja if tiene_caja else np.nan,
                  r.freq_alta if tiene_caja else np.nan) for i, r in g.iterrows()]
        tareas.append((indice[audio], filas))
    if faltan:
        rep(f"{faltan} pulsos sin audio encontrado: quedan sin frecuencia", "AVISO")

    procesos = P.get("procesos") or os.cpu_count()
    rep(f"Midiendo frecuencia pico en {len(tareas)} audios con {procesos} procesos "
        f"({'caja anotada' if tiene_caja else 'banda fija'} {F['banda_hz'][0]}-{F['banda_hz'][1]} Hz)...")
    resultados = []
    with mp.Pool(processes=procesos) as pool:
        for r in pool.imap_unordered(frecuencias_de_un_audio, tareas):
            resultados.extend(r)
    res = pd.DataFrame(resultados, columns=["idx", "sample_rate_hz", "freq_pico_hz", "nivel_pico_db"]).set_index("idx")
    return pulsos.join(res)


def clasificar(pulsos):
    umbral = F["umbral_transicion_hz"]
    max_sin = P["intervalos"]["max_intervalo_sin_secuencia_s"]
    g = pulsos.groupby("archivo", sort=False)
    f_ant = g["freq_pico_hz"].shift(1)
    sec_ant = g["secuencia_id"].shift(1)
    pulsos["freq_anterior_hz"] = f_ant
    pulsos["delta_freq_hz"] = pulsos["freq_pico_hz"] - f_ant
    pulsos["transicion"] = np.select(
        [pulsos["delta_freq_hz"] > umbral, pulsos["delta_freq_hz"] < -umbral, pulsos["delta_freq_hz"].notna()],
        ["sube", "baja", "igual"], default="sin_dato")

    base = pulsos["intervalo_anterior_s"].notna() & pulsos["misma_especie_anterior"].astype(bool) \
        & ~pulsos["solapa_anterior"].astype(bool)
    en_secuencia = base & pulsos["misma_secuencia_anterior"].astype(bool)
    sin_secuencia = base & pulsos["secuencia_id"].isna() & sec_ant.isna() \
        & (pulsos["intervalo_anterior_s"] <= max_sin)
    pulsos["fuente_intervalo"] = np.select([en_secuencia, sin_secuencia], ["secuencia", "sin_secuencia"],
                                           default="no_valido")
    pulsos["intervalo_valido"] = pulsos["fuente_intervalo"] != "no_valido"
    return pulsos


def estadisticas(validos):
    pcts = P["intervalos"]["percentiles"]
    filas = []
    grupos = [(esp, "todas", g) for esp, g in validos.groupby("especie")]
    grupos += [(esp, tr, g) for (esp, tr), g in validos.groupby(["especie", "transicion"])]
    for esp, tr, g in grupos:
        ms = g["intervalo_anterior_s"].to_numpy() * 1000
        fila = {"especie": esp, "transicion": tr, "n": len(ms), "audios": g["archivo"].nunique()}
        for p, v in zip(pcts, np.percentile(ms, pcts)):
            fila[f"p{p}_ms"] = round(float(v), 2)
        fila["media_ms"] = round(float(ms.mean()), 2)
        fila["cv"] = round(float(ms.std() / ms.mean()), 3) if ms.mean() > 0 else np.nan
        filas.append(fila)
    orden = {"todas": 0, "sube": 1, "baja": 2, "igual": 3, "sin_dato": 4}
    tabla = pd.DataFrame(filas)
    tabla["_orden"] = tabla["transicion"].map(orden)
    return tabla.sort_values(["especie", "_orden"]).drop(columns="_orden")


def frecuencias(pulsos, validos):
    filas = []
    for esp, g in pulsos.groupby("especie"):
        v = validos[validos["especie"] == esp]
        cambios = v["transicion"].isin(["sube", "baja"]).sum()
        filas.append({
            "especie": esp,
            "pulsos_con_freq": int(g["freq_pico_hz"].notna().sum()),
            "freq_p5_khz": round(g["freq_pico_hz"].quantile(0.05) / 1000, 2),
            "freq_p50_khz": round(g["freq_pico_hz"].quantile(0.50) / 1000, 2),
            "freq_p95_khz": round(g["freq_pico_hz"].quantile(0.95) / 1000, 2),
            "pares_validos": len(v),
            "pct_pares_con_cambio_freq": round(100 * cambios / len(v), 1) if len(v) else np.nan,
            "delta_abs_p50_khz": round(v["delta_freq_hz"].abs().median() / 1000, 2) if len(v) else np.nan,
        })
    return pd.DataFrame(filas)


def histogramas(validos, carpeta):
    carpeta.mkdir(parents=True, exist_ok=True)
    colores = {"sube": "#2a78c8", "baja": "#d9822b", "igual": "#7a7a7a"}
    for esp, g in validos.groupby("especie"):
        ms = g["intervalo_anterior_s"] * 1000
        if len(ms) < 2:
            continue
        bins = np.linspace(0, ms.quantile(0.99) * 1.1, 60)
        fig, ax = plt.subplots(figsize=(7, 3.2))
        datos = [ms[g["transicion"] == t] for t in colores]
        ax.hist(datos, bins=bins, stacked=True, color=list(colores.values()), label=list(colores))
        for p in (5, 50, 95):
            ax.axvline(np.percentile(ms, p), color="black", lw=0.8, ls="--")
        ax.set_xlabel("intervalo entre inicios de pulsos consecutivos (ms)")
        ax.set_ylabel("pares")
        ax.set_title(f"{esp}  (n={len(ms)}; lineas: p5, p50, p95)", fontsize=10)
        ax.legend(title="frecuencia", fontsize=8, title_fontsize=8)
        fig.tight_layout()
        fig.savefig(carpeta / f"{esp}.png", dpi=120)
        plt.close(fig)


def alineacion_con_raven(pulsos, rep):
    """
    Compara la frecuencia pico medida aqui con la que Raven guardo al anotar.
    Si coinciden, el recorte de audio cae sobre el pulso anotado: confirma que
    Begin Time es relativo al archivo de Begin File y que el audio es el correcto.
    """
    if "freq_pico_raven" not in pulsos.columns:
        return
    rep.titulo("Alineacion: frecuencia pico medida vs. Peak Freq de Raven")
    d = pulsos.dropna(subset=["freq_pico_hz", "freq_pico_raven"]).copy()
    if d.empty:
        rep("sin pulsos con ambas frecuencias", "AVISO")
        return
    d["dif_khz"] = (d["freq_pico_hz"] - d["freq_pico_raven"]).abs() / 1000
    umbral = F["umbral_transicion_hz"] / 1000
    tabla = d.groupby("especie").agg(pulsos=("dif_khz", "size"),
                                     dif_mediana_khz=("dif_khz", "median"),
                                     pct_a_menos_de_umbral=("dif_khz", lambda x: round(100 * (x < umbral).mean(), 1)))
    tabla["dif_mediana_khz"] = tabla["dif_mediana_khz"].round(2)
    rep(tabla.to_string())
    pct = 100 * (d["dif_khz"] < umbral).mean()
    rep(f"{pct:.1f} % de los pulsos con diferencia < {umbral:g} kHz",
        "OK" if pct >= 80 else "AVISO")
    if pct < 80:
        rep("diferencias grandes: revisar si Begin Time es relativo a Begin File, si el audio indexado "
            "es el correcto (nombres repetidos en carpetas distintas) o si Raven midio en otra banda", "AVISO")


def contrastar(stats, rep):
    refs = P.get("referencias") or {}
    todas = stats[stats["transicion"] == "todas"].set_index("especie")
    rep.titulo("Contraste con rangos publicados (mediana de todas las transiciones)")
    for esp, ref in refs.items():
        if esp not in todas.index:
            rep(f"{esp}: sin pares validos en los datos", "AVISO")
            continue
        p50 = todas.at[esp, "p50_ms"]
        dentro = ref.get("min_ms", 0) <= p50 <= ref.get("max_ms", float("inf"))
        if "min_ms" in ref and "max_ms" in ref:
            rango = f"{ref['min_ms']}-{ref['max_ms']} ms"
        elif "max_ms" in ref:
            rango = f"< {ref['max_ms']} ms"
        else:
            rango = f"> {ref['min_ms']} ms"
        rep(f"{esp}: p50 = {p50} ms, rango publicado {rango} ({ref['fuente']})", "OK" if dentro else "AVISO")


def main():
    rep = Reporte(rutas.SALIDA_P2 / "intervalos.txt")
    if not rutas.ARCHIVO_PULSOS_VINCULADOS.exists():
        raise SystemExit(f"Falta {rutas.ARCHIVO_PULSOS_VINCULADOS}: correr antes p1_vincular.py")
    pulsos = pd.read_csv(rutas.ARCHIVO_PULSOS_VINCULADOS)
    for c in ("freq_baja", "freq_alta"):
        if c in pulsos.columns:
            pulsos[c] = pd.to_numeric(pulsos[c], errors="coerce")

    pulsos = medir_frecuencias(pulsos, rep)
    pulsos = clasificar(pulsos)
    rutas.SALIDA_P2.mkdir(parents=True, exist_ok=True)
    pulsos.to_csv(rutas.ARCHIVO_PULSOS_FRECUENCIA, index=False)

    validos = pulsos[pulsos["intervalo_valido"]]
    if validos.empty:
        rep("no hay pares de pulsos consecutivos validos", "ERROR")
        rep.guardar()
        raise SystemExit(1)

    stats = estadisticas(validos)
    stats.to_csv(rutas.SALIDA_P2 / "intervalos_por_especie.csv", index=False)
    freqs = frecuencias(pulsos, validos)
    freqs.to_csv(rutas.SALIDA_P2 / "frecuencias_por_especie.csv", index=False)
    histogramas(validos, rutas.SALIDA_P2 / "histogramas")

    rep.titulo("Origen de los pares validos")
    rep(validos.groupby(["especie", "fuente_intervalo"]).size().unstack(fill_value=0).to_string())
    rep.titulo("Intervalos por especie y transicion (ms)")
    rep(stats.to_string(index=False))
    rep.titulo("Frecuencia pico y alternancia")
    rep(freqs.to_string(index=False))
    alineacion_con_raven(pulsos, rep)
    contrastar(stats, rep)
    rep(f"\nSalidas en {rutas.SALIDA_P2}")
    rep.guardar()


if __name__ == "__main__":
    main()

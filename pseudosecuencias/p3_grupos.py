"""
P3 - Formar pseudo-secuencias de 2 y 3 pulsos, fijar la ventana unica y partir por grabacion.

    python p3_grupos.py

1. Grupos: n pulsos CONSECUTIVOS del mismo audio (orden de P1, con todos los
   pulsos) donde cada enlace es valido (P2: misma especie, sin solapamiento,
   misma secuencia anotada) y cada pulso tiene una calidad aceptada. El
   intervalo NO se filtra por percentiles salvo que se active
   grupos.filtrar_por_percentil: la distancia entre pulsos, con su
   variabilidad, es parte de lo que el modelo debe aprender.
2. Ventana unica: todas las imagenes, de 2 y de 3 pulsos, cubren la MISMA
   duracion con el MISMO ancho en pixeles. La duracion sale del cuantil de la
   duracion de los grupos mas largos (+ margenes); los que no caben se excluyen.
3. Particion por grabacion: todos los grupos de un audio van a la misma
   particion (los grupos de un audio comparten pulsos: partir por imagen
   filtraria pulsos de entrenamiento a la prueba).

Salidas en <SALIDA>/P3_grupos/:
  grupos.csv    todos los grupos candidatos, con 'incluido', 'motivo' y 'particion'
  ventana.json  geometria comun (sr, hop, ms por pixel, alto x ancho, banda)
  grupos.txt    reporte
"""

import json

import numpy as np
import pandas as pd
import soundfile as sf

import render
import rutas
from comun import Reporte

P = rutas.PARAMETROS
G, V, R = P["grupos"], P["ventana"], P["render"]


def rangos_de_intervalo(rep):
    """{(especie, transicion): (min_ms, max_ms)} con respaldo a 'todas' si hay pocos pares."""
    stats = pd.read_csv(rutas.ARCHIVO_INTERVALOS)
    cb, ca = f"p{G['percentil_bajo']}_ms", f"p{G['percentil_alto']}_ms"
    if cb not in stats.columns or ca not in stats.columns:
        raise SystemExit(f"intervalos_por_especie.csv no tiene {cb} o {ca}: agregar esos percentiles "
                         f"en intervalos.percentiles de parametros.yaml y volver a correr p2_intervalos.py")
    todas = stats[stats["transicion"] == "todas"].set_index("especie")
    rangos = {}
    for _, f in stats.iterrows():
        if f["transicion"] == "todas":
            continue
        if f["n"] >= G["min_pares_por_transicion"]:
            rangos[(f["especie"], f["transicion"])] = (f[cb], f[ca])
        else:
            t = todas.loc[f["especie"]]
            rangos[(f["especie"], f["transicion"])] = (t[cb], t[ca])
            rep(f"{f['especie']}/{f['transicion']}: solo {int(f['n'])} pares, se usa el rango de 'todas'", "AVISO")
    return rangos


def patron_de(transiciones):
    """
    Patron de frecuencia del grupo a partir de las transiciones entre pulsos:
      2 pulsos: baja -> alto-bajo, sube -> bajo-alto
      3 pulsos: baja;sube -> alto-bajo-alto, sube;baja -> bajo-alto-bajo
      todas 'igual' -> constante; cualquier otra combinacion -> irregular
    (generaliza a n pulsos: alternancia estricta = alto-bajo-alto-... o bajo-alto-bajo-...)
    """
    t = list(transiciones)
    if all(x == "igual" for x in t):
        return "constante"
    alterna = all(x in ("sube", "baja") for x in t) and all(a != b for a, b in zip(t, t[1:]))
    if not alterna:
        return "irregular"
    nivel = "alto" if t[0] == "baja" else "bajo"
    niveles = [nivel]
    for _ in t:
        nivel = "bajo" if nivel == "alto" else "alto"
        niveles.append(nivel)
    return "-".join(niveles)


def formar_grupos(pulsos, rangos):
    filas = []
    calidades = set(G["calidades"])
    for audio, g in pulsos.groupby("archivo", sort=False):
        g = g.sort_values("orden_en_audio").reset_index(drop=True)
        n_p = len(g)
        # enlace_ok[k]: el pulso k se une validamente con el k-1
        intervalo_ms = g["intervalo_anterior_s"] * 1000
        # El intervalo es parte de lo que el modelo debe aprender, con su variabilidad:
        # por defecto solo se exige que el enlace sea valido (P2). El filtro por
        # percentiles es opcional (grupos.filtrar_por_percentil).
        if rangos is None:
            en_rango = [bool(g.at[k, "intervalo_valido"]) for k in range(n_p)]
        else:
            en_rango = [
                bool(g.at[k, "intervalo_valido"]) and (g.at[k, "especie"], g.at[k, "transicion"]) in rangos
                and rangos[(g.at[k, "especie"], g.at[k, "transicion"])][0] <= intervalo_ms[k]
                <= rangos[(g.at[k, "especie"], g.at[k, "transicion"])][1]
                for k in range(n_p)
            ]
        calidad_ok = g["calidad"].isin(calidades).to_numpy() & g["freq_pico_hz"].notna().to_numpy()
        for n in G["tamanos"]:
            ultimo_inicio = -10 ** 9
            for i in range(n_p - n + 1):
                idx = range(i, i + n)
                motivo = None
                if not all(en_rango[k] for k in range(i + 1, i + n)):
                    continue   # no son consecutivos validos: no es un grupo candidato
                if not all(calidad_ok[k] for k in idx):
                    motivo = "calidad"
                elif i - ultimo_inicio < G["paso"]:
                    motivo = "paso"
                if motivo is None:
                    ultimo_inicio = i
                sub = g.iloc[i:i + n]
                filas.append({
                    "archivo": audio,
                    "especie": sub["especie"].iloc[0],
                    "n": n,
                    "ids": ";".join(map(str, sub["id"])),
                    "inicio_s": sub["inicio"].iloc[0],
                    "fin_s": sub["fin"].iloc[-1],
                    "duracion_ms": round((sub["fin"].iloc[-1] - sub["inicio"].iloc[0]) * 1000, 3),
                    "inicios_s": ";".join(f"{v:.6f}" for v in sub["inicio"]),
                    "fines_s": ";".join(f"{v:.6f}" for v in sub["fin"]),
                    "intervalos_ms": ";".join(f"{v:.2f}" for v in intervalo_ms.iloc[i + 1:i + n]),
                    "transiciones": ";".join(sub["transicion"].iloc[1:]),
                    "patron": patron_de(sub["transicion"].iloc[1:]),
                    "freqs_khz": ";".join(f"{v / 1000:.2f}" for v in sub["freq_pico_hz"]),
                    "calidades": ";".join(map(str, sub["calidad"])),
                    "secuencia_id": sub["secuencia_id"].iloc[0],
                    "motivo": motivo,
                })
    return pd.DataFrame(filas)


def fijar_ventana(grupos, rep):
    geo = render.geometria(R, P["frecuencia"]["banda_hz"])
    ancho, margen = V["ancho_px"], V["margen_ms"]
    if ancho % 32:
        raise SystemExit("ventana.ancho_px debe ser multiplo de 32")
    if V.get("duracion_ms"):
        necesaria, origen = float(V["duracion_ms"]), "fijada a mano"
    else:
        n_max = max(G["tamanos"])
        base = grupos[(grupos["n"] == n_max) & grupos["motivo"].isna()]["duracion_ms"]
        if base.empty:
            raise SystemExit(f"No hay grupos validos de {n_max} pulsos para fijar la ventana")
        q = V["cuantil_duracion"]
        necesaria, origen = float(base.quantile(q)) + 2 * margen, f"cuantil {q} de grupos de {n_max} pulsos + 2 margenes"
    hop = render.hop_para(necesaria, ancho, geo["sr"], geo["n_fft"])
    duracion = render.duracion_efectiva_ms(hop, ancho, geo["sr"], geo["n_fft"])
    ventana = {
        **geo,
        "banda_hz": P["frecuencia"]["banda_hz"],
        "ancho_px": ancho,
        "hop": hop,
        "ms_por_px": hop / geo["sr"] * 1000,
        "duracion_ms": duracion,
        "duracion_necesaria_ms": necesaria,
        "origen_duracion": origen,
        "margen_ms": margen,
        "rango_db": R["rango_db"],
        "colormap": R["colormap"],
    }
    rep(f"ventana: {geo['alto_px']} x {ancho} px (alto x ancho), {duracion:.1f} ms ({origen}), "
        f"{ventana['ms_por_px']:.4f} ms por columna, {geo['hz_por_fila']:.0f} Hz por fila", "OK")
    return ventana


def marcar_exclusiones(grupos, ventana, rep):
    duraciones = {}
    indice = rutas.indexar_audios(str(rutas.RUTA_AUDIOS))
    for a in grupos["archivo"].unique():
        try:
            duraciones[a] = sf.info(indice[a]).duration
        except Exception:
            duraciones[a] = np.nan
    m = ventana["margen_ms"] / 1000
    libres = grupos["motivo"].isna()
    no_cabe = libres & (grupos["duracion_ms"] + 2 * ventana["margen_ms"] > ventana["duracion_ms"])
    grupos.loc[no_cabe, "motivo"] = "no_cabe_en_ventana"
    dur = grupos["archivo"].map(duraciones)
    borde = grupos["motivo"].isna() & ((grupos["inicio_s"] - m < 0) | (grupos["fin_s"] + m > dur) | dur.isna())
    grupos.loc[borde, "motivo"] = "borde_o_audio_no_disponible"
    grupos["incluido"] = grupos["motivo"].isna()
    return grupos


def particionar(grupos, rep):
    props = P["particion"]["proporciones"]
    nombres = ["train", "val", "test"]
    rng = np.random.default_rng(P["particion"]["semilla"])
    inc = grupos[grupos["incluido"]]
    por_audio = inc.groupby("archivo").agg(especie=("especie", lambda s: s.value_counts().index[0]),
                                           grupos=("n", "size"))
    asignacion = {}
    for esp, aud in por_audio.groupby("especie"):
        # orden aleatorio (semilla fija) y, a igualdad, las grabaciones grandes primero;
        # cada grabacion va a la particion que mas grupos le faltan para su proporcion
        orden = list(aud.index)
        rng.shuffle(orden)
        orden.sort(key=lambda a: -aud.at[a, "grupos"])
        total = aud["grupos"].sum()
        asignados = dict.fromkeys(nombres, 0)
        for a in orden:
            deficit = {s: p * total - asignados[s] for s, p in zip(nombres, props)}
            elegido = max(nombres, key=lambda s: deficit[s])
            asignacion[a] = elegido
            asignados[elegido] += aud.at[a, "grupos"]
        faltan = [s for s in nombres if s not in {asignacion[a] for a in orden}]
        if faltan:
            rep(f"{esp}: {len(orden)} audio(s) con grupos; sin grabaciones para {faltan}", "AVISO")
    grupos["particion"] = grupos["archivo"].map(asignacion).where(grupos["incluido"])
    return grupos


def main():
    rep = Reporte(rutas.SALIDA_P3 / "grupos.txt")
    for req in (rutas.ARCHIVO_PULSOS_FRECUENCIA, rutas.ARCHIVO_INTERVALOS):
        if not req.exists():
            raise SystemExit(f"Falta {req}: correr antes p2_intervalos.py")
    pulsos = pd.read_csv(rutas.ARCHIVO_PULSOS_FRECUENCIA)
    if G.get("filtrar_por_percentil", False):
        rangos = rangos_de_intervalo(rep)
        rep(f"filtro de intervalos activo: p{G['percentil_bajo']}-p{G['percentil_alto']} por especie y transicion", "AVISO")
    else:
        rangos = None
        rep("sin filtro de intervalos por percentil: se conserva toda la variabilidad del espaciado "
            "(solo se excluyen enlaces no validos y grupos que no caben en la ventana)", "OK")

    grupos = formar_grupos(pulsos, rangos)
    if grupos.empty:
        rep("no se formo ningun grupo", "ERROR")
        rep.guardar()
        raise SystemExit(1)
    ventana = fijar_ventana(grupos, rep)
    grupos = marcar_exclusiones(grupos, ventana, rep)
    grupos = particionar(grupos, rep)

    rutas.SALIDA_P3.mkdir(parents=True, exist_ok=True)
    grupos.to_csv(rutas.ARCHIVO_GRUPOS, index=False)
    with open(rutas.ARCHIVO_VENTANA, "w", encoding="utf-8") as f:
        json.dump(ventana, f, indent=2)

    rep.titulo("Grupos candidatos y exclusiones")
    rep(grupos.assign(motivo=grupos["motivo"].fillna("incluido"))
        .groupby(["especie", "n", "motivo"]).size().unstack(fill_value=0).to_string())

    inc = grupos[grupos["incluido"]]
    rep.titulo("Grupos incluidos por especie, n y particion")
    rep(inc.groupby(["especie", "n", "particion"]).size().unstack(fill_value=0).to_string())
    rep.titulo("Patrones de frecuencia incluidos por especie y n (condicion del modelo)")
    rep(inc.groupby(["especie", "n", "patron"]).size().unstack(fill_value=0).to_string())
    rep.titulo("Grabaciones por particion")
    rep(inc.groupby(["especie", "particion"])["archivo"].nunique().unstack(fill_value=0).to_string())

    rep.titulo("Tamano de los pulsos y del silencio en la imagen")
    msp = ventana["ms_por_px"]
    filas = []
    for esp, g in inc.groupby("especie"):
        dur_pulso = [float(b) - float(a) for i, f in zip(g["inicios_s"], g["fines_s"])
                     for a, b in zip(i.split(";"), f.split(";"))]
        intervalos = [float(v) for s in g["intervalos_ms"] for v in s.split(";")]
        filas.append({"especie": esp,
                      "pulso_px_p50": round(np.median(dur_pulso) * 1000 / msp, 1),
                      "intervalo_px_p5": round(np.percentile(intervalos, 5) / msp, 1),
                      "intervalo_px_p50": round(np.median(intervalos) / msp, 1),
                      "grupo_px_p95": round((g["duracion_ms"].quantile(0.95) + 2 * ventana["margen_ms"]) / msp, 1)})
    tabla = pd.DataFrame(filas)
    rep(tabla.to_string(index=False))
    if (tabla["pulso_px_p50"] < 8).any():
        rep("hay especies con pulsos de menos de 8 columnas: considerar ancho_px 1024 o un cuantil menor", "AVISO")
    rep(f"\nTotal: {len(inc)} grupos incluidos de {len(grupos)} candidatos")
    rep(f"Salidas: {rutas.ARCHIVO_GRUPOS} y {rutas.ARCHIVO_VENTANA}")
    rep.guardar()


if __name__ == "__main__":
    main()

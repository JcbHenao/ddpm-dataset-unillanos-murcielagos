"""
P1 - Vincular pulsos con su audio, su orden y su secuencia.

    python p1_vincular.py

Para cada audio ordena TODOS los pulsos por tiempo (antes de cualquier filtro
de calidad: si se descarta un pulso intermedio, el intervalo entre sus vecinos
seria falso), asigna cada pulso a la secuencia anotada que lo contiene y calcula
la relacion con el pulso anterior.

Salidas en <SALIDA>/P1_vinculacion/:
  pulsos_vinculados.csv   una fila por pulso
  resumen_vinculacion.csv una fila por especie
  vinculacion.txt         reporte
"""

import numpy as np
import pandas as pd

import rutas
from comun import Reporte, cargar_maestro, separar


def asignar_secuencias(pulsos, secuencias, tolerancia):
    """Para cada pulso, la secuencia del mismo audio que contiene su centro (prefiere misma especie)."""
    sec_id = pd.Series(pd.NA, index=pulsos.index, dtype="object")
    sec_esp = pd.Series(pd.NA, index=pulsos.index, dtype="object")
    n_candidatas = pd.Series(0, index=pulsos.index)
    por_audio = {a: g for a, g in secuencias.groupby("archivo")}
    for audio, grupo in pulsos.groupby("archivo"):
        s = por_audio.get(audio)
        if s is None:
            continue
        centros = ((grupo["inicio"] + grupo["fin"]) / 2).to_numpy()
        ini = s["inicio"].to_numpy() - tolerancia
        fin = s["fin"].to_numpy() + tolerancia
        dentro = (centros[:, None] >= ini[None, :]) & (centros[:, None] <= fin[None, :])
        for fila, idx in enumerate(grupo.index):
            cand = np.flatnonzero(dentro[fila])
            n_candidatas[idx] = len(cand)
            if not len(cand):
                continue
            misma = [c for c in cand if s["especie"].iloc[c] == pulsos.at[idx, "especie"]]
            elegida = misma[0] if misma else cand[0]
            sec_id[idx] = s["id"].iloc[elegida]
            sec_esp[idx] = s["especie"].iloc[elegida]
    return sec_id, sec_esp, n_candidatas


RANGO_CALIDAD = {"Alta": 0, "Media": 1, "Baja": 2}


def quitar_dobles_anotacion(pulsos, umbral):
    """
    Dos anotadores pueden marcar el mismo pulso con tiempos casi iguales; la fase 2
    del pipeline solo quita duplicados exactos. Si dos pulsos del mismo audio y
    especie, de anotadores distintos, se solapan >= umbral (interseccion / union),
    se conserva el de mejor calidad (a igualdad, el primero).
    """
    if "anotador" not in pulsos.columns:
        return pulsos, pulsos.iloc[0:0]
    quitar = set()
    for _, g in pulsos.groupby("archivo"):
        activos = []
        for idx in g.sort_values("inicio").index:
            r = pulsos.loc[idx]
            activos = [k for k in activos if pulsos.at[k, "fin"] > r["inicio"]]
            doble = False
            for k in activos:
                a = pulsos.loc[k]
                if a["especie"] != r["especie"] or a["anotador"] == r["anotador"]:
                    continue
                inter = min(a["fin"], r["fin"]) - max(a["inicio"], r["inicio"])
                union = max(a["fin"], r["fin"]) - min(a["inicio"], r["inicio"])
                if union > 0 and inter / union >= umbral:
                    if RANGO_CALIDAD.get(r["calidad"], 3) < RANGO_CALIDAD.get(a["calidad"], 3):
                        quitar.add(k)
                        activos.remove(k)
                        activos.append(idx)
                    else:
                        quitar.add(idx)
                    doble = True
                    break
            if not doble:
                activos.append(idx)
    return pulsos.drop(index=sorted(quitar)), pulsos.loc[sorted(quitar)]


def main():
    rep = Reporte(rutas.SALIDA_P1 / "vinculacion.txt")
    tol = rutas.PARAMETROS["vinculacion"]["tolerancia_s"]

    df = cargar_maestro()
    pulsos, secuencias = separar(df)
    validos = pulsos["inicio"].notna() & pulsos["fin"].notna() & (pulsos["fin"] > pulsos["inicio"])
    if (~validos).any():
        rep(f"{(~validos).sum()} pulsos con tiempos invalidos se descartan", "AVISO")
    pulsos = pulsos[validos].copy()

    duplicados = pulsos.duplicated(subset=["archivo", "inicio", "fin"], keep="first")
    if duplicados.any():
        rep(f"{duplicados.sum()} pulsos duplicados (mismo audio, inicio y fin) se descartan", "AVISO")
    pulsos = pulsos[~duplicados]

    pulsos, dobles = quitar_dobles_anotacion(pulsos, rutas.PARAMETROS["vinculacion"]["solape_mismo_pulso"])
    if len(dobles):
        rutas.SALIDA_P1.mkdir(parents=True, exist_ok=True)
        dobles.to_csv(rutas.SALIDA_P1 / "pulsos_doble_anotacion.csv", index=False)
        rep(f"{len(dobles)} pulsos anotados dos veces por anotadores distintos: se conserva uno "
            f"(los descartados quedan en pulsos_doble_anotacion.csv)", "AVISO")

    pulsos = pulsos.sort_values(["archivo", "inicio", "fin"]).reset_index(drop=True)
    pulsos["secuencia_id"], pulsos["secuencia_especie"], pulsos["n_secuencias_candidatas"] = \
        asignar_secuencias(pulsos, secuencias, tol)
    pulsos["especie_coincide_con_secuencia"] = pulsos["secuencia_especie"].isna() | \
        (pulsos["secuencia_especie"] == pulsos["especie"])

    g = pulsos.groupby("archivo", sort=False)
    pulsos["orden_en_audio"] = g.cumcount()
    anterior = g.shift(1)
    pulsos["id_anterior"] = anterior["id"]
    pulsos["intervalo_anterior_s"] = pulsos["inicio"] - anterior["inicio"]      # inicio a inicio
    pulsos["silencio_anterior_s"] = pulsos["inicio"] - anterior["fin"]          # fin del anterior a inicio
    pulsos["solapa_anterior"] = pulsos["silencio_anterior_s"] < 0
    pulsos["misma_especie_anterior"] = pulsos["especie"] == anterior["especie"]
    pulsos["misma_secuencia_anterior"] = pulsos["secuencia_id"].notna() & \
        (pulsos["secuencia_id"] == anterior["secuencia_id"])
    pulsos["duracion_s"] = pulsos["fin"] - pulsos["inicio"]

    columnas = ["id", "archivo", "especie", "calidad", "tipo_pulso", "inicio", "fin", "duracion_s",
                "orden_en_audio", "secuencia_id", "secuencia_especie", "n_secuencias_candidatas",
                "especie_coincide_con_secuencia", "id_anterior", "intervalo_anterior_s",
                "silencio_anterior_s", "solapa_anterior", "misma_especie_anterior", "misma_secuencia_anterior"]
    columnas += [c for c in ("freq_baja", "freq_alta", "freq_pico_raven", "anotador") if c in pulsos.columns]
    rutas.SALIDA_P1.mkdir(parents=True, exist_ok=True)
    pulsos[columnas].to_csv(rutas.ARCHIVO_PULSOS_VINCULADOS, index=False)

    filas = []
    for esp, p in pulsos.groupby("especie"):
        filas.append({
            "especie": esp,
            "pulsos": len(p),
            "audios": p["archivo"].nunique(),
            "en_secuencia": int(p["secuencia_id"].notna().sum()),
            "sin_secuencia": int(p["secuencia_id"].isna().sum()),
            "secuencias_distintas": p["secuencia_id"].nunique(),
            "en_mas_de_una_secuencia": int((p["n_secuencias_candidatas"] > 1).sum()),
            "especie_distinta_a_su_secuencia": int((~p["especie_coincide_con_secuencia"]).sum()),
            "consecutivos_misma_secuencia": int(p["misma_secuencia_anterior"].sum()),
            "solapados_con_anterior": int(p["solapa_anterior"].sum()),
            "pulsos_alta": int((p["calidad"] == "Alta").sum()),
        })
    resumen = pd.DataFrame(filas)
    resumen.to_csv(rutas.SALIDA_P1 / "resumen_vinculacion.csv", index=False)

    rep.titulo("Vinculacion por especie")
    rep(resumen.to_string(index=False))
    rep("")
    rep(f"{len(pulsos)} pulsos en {pulsos['archivo'].nunique()} audios", "OK")
    sin = pulsos["secuencia_id"].isna().mean() * 100
    rep(f"{sin:.1f} % de pulsos fuera de toda secuencia anotada",
        "AVISO" if sin > 20 else "OK")
    if (~pulsos["especie_coincide_con_secuencia"]).any():
        rep("hay pulsos cuya especie difiere de la de su secuencia (ver especie_distinta_a_su_secuencia): "
            "posibles dos murcielagos en el mismo audio o error de anotacion", "AVISO")
    if pulsos["solapa_anterior"].any():
        rep("hay pulsos que se solapan con el anterior: no se usaran como consecutivos", "AVISO")
    rep(f"\nSalida: {rutas.ARCHIVO_PULSOS_VINCULADOS}")
    rep.guardar()


if __name__ == "__main__":
    main()

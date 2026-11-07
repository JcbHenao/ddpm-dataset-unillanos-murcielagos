"""
P0 - Verificar entorno y rutas ANTES de procesar nada.

    python p0_verificar.py

Revisa: paquetes del entorno, que config.py del pipeline resuelva las rutas,
columnas y valores del archivo maestro, audios encontrados y su frecuencia de
muestreo (Nyquist contra la banda de trabajo), y que la carpeta de salida
sea escribible. No modifica nada del pipeline.
Reporte: <SALIDA>/P0_verificacion/verificacion.txt
"""

import sys
from collections import Counter
from importlib.metadata import PackageNotFoundError, version

import rutas
from comun import Reporte, cargar_maestro, separar


def main():
    rep = Reporte(rutas.SALIDA_P0 / "verificacion.txt")

    rep.titulo("Entorno")
    rep(f"Python {sys.version.split()[0]} en {sys.executable}", "OK")
    if sys.prefix == sys.base_prefix:
        rep("no se esta usando un entorno virtual (activar .venv-pseudo)", "AVISO")
    for paquete in ["numpy", "pandas", "librosa", "soundfile", "matplotlib", "pillow", "pyyaml", "scipy", "tqdm"]:
        try:
            rep(f"{paquete} {version(paquete)}", "OK")
        except PackageNotFoundError:
            rep(f"falta {paquete} (pip install -r requirements.txt)", "ERROR")

    rep.titulo("Rutas (leidas de config.py del pipeline)")
    for nombre, ruta in rutas.resumen().items():
        existe = ruta.exists()
        nivel = "OK" if existe or nombre.startswith("SALIDA") else "ERROR"
        rep(f"{nombre}: {ruta}{'' if existe else '  (no existe)'}", nivel)
    try:
        rutas.SALIDA_P0.mkdir(parents=True, exist_ok=True)
        prueba = rutas.SALIDA_P0 / ".escritura"
        prueba.write_text("ok")
        prueba.unlink()
        rep("la carpeta de salida es escribible", "OK")
    except OSError as e:
        rep(f"no se puede escribir en {rutas.SALIDA}: {e}", "ERROR")

    if not rutas.ARCHIVO_MAESTRO.exists():
        rep("sin archivo maestro no se puede seguir", "ERROR")
        rep.guardar()
        sys.exit(1)

    rep.titulo("Archivo maestro")
    df = cargar_maestro()
    pulsos, secuencias = separar(df)
    rep(f"{len(df)} anotaciones: {len(pulsos)} pulsos, {len(secuencias)} secuencias", "OK")
    otros = Counter(df["tipo"]) - Counter({rutas.PARAMETROS['valores_tipo']['pulso']: len(pulsos),
                                          rutas.PARAMETROS['valores_tipo']['secuencia']: len(secuencias)})
    if otros:
        rep(f"otros valores de tipo (se ignoran): {dict(otros)}", "AVISO")
    for c, sin in (("freq_baja", "se usa la banda fija"), ("freq_alta", "se usa la banda fija"),
                   ("freq_pico_raven", "no se puede comprobar la alineacion de tiempos"),
                   ("anotador", "no se detectan pulsos anotados dos veces")):
        rep(f"columna opcional {c}: {'presente' if c in df.columns else 'ausente (' + sin + ')'}",
            "OK" if c in df.columns else "AVISO")
    if "anotador" in df.columns:
        rep(f"pulsos por anotador: {dict(Counter(pulsos['anotador'].fillna('SinDato')))}", "OK")
        dobles = pulsos.groupby("archivo")["anotador"].nunique()
        n = int((dobles > 1).sum())
        rep(f"{n} audios con pulsos de mas de un anotador (P1 elimina el mismo pulso anotado dos veces)",
            "AVISO" if n else "OK")

    malos = df[df["inicio"].isna() | df["fin"].isna() | (df["fin"] <= df["inicio"])]
    rep(f"{len(malos)} anotaciones con tiempos vacios o fin <= inicio",
        "AVISO" if len(malos) else "OK")

    rep("")
    rep(f"{'especie':<20}{'pulsos':>8}{'secuencias':>12}{'audios':>8}  calidades de pulsos")
    for esp in sorted(df["especie"].unique()):
        p = pulsos[pulsos["especie"] == esp]
        s = secuencias[secuencias["especie"] == esp]
        cal = dict(Counter(p["calidad"].fillna("SinDato")))
        rep(f"{esp:<20}{len(p):>8}{len(s):>12}{p['archivo'].nunique():>8}  {cal}")

    rep.titulo("Audios")
    indice = rutas.indexar_audios(str(rutas.RUTA_AUDIOS))
    rep(f"{len(indice)} archivos .wav indexados en {rutas.RUTA_AUDIOS}", "OK" if indice else "ERROR")
    # indexar_audios del pipeline indexa por NOMBRE de archivo: dos .wav con el mismo
    # nombre en carpetas distintas se pisan sin aviso (se usaria solo el ultimo)
    import os
    nombres = Counter(f for _, _, fs in os.walk(rutas.RUTA_AUDIOS) for f in fs if f.lower().endswith(".wav"))
    repetidos = {n: c for n, c in nombres.items() if c > 1}
    rep(f"{len(repetidos)} nombres de .wav repetidos en carpetas distintas",
        "AVISO" if repetidos else "OK")
    for n, c in list(repetidos.items())[:10]:
        rep(f"  {n}: {c} copias (se usa solo una)")
    archivos = set(pulsos["archivo"].dropna())
    faltan = sorted(archivos - set(indice))
    rep(f"{len(archivos) - len(faltan)} de {len(archivos)} audios con pulsos encontrados",
        "OK" if not faltan else "AVISO")
    for a in faltan[:10]:
        rep(f"  no encontrado: {a}")
    if len(faltan) > 10:
        rep(f"  ... y {len(faltan) - 10} mas")

    import soundfile as sf
    banda_alta = rutas.PARAMETROS["frecuencia"]["banda_hz"][1]
    tasas, bajos = Counter(), []
    for a in sorted(archivos & set(indice)):
        try:
            sr = sf.info(indice[a]).samplerate
        except Exception as e:
            rep(f"no se pudo leer {a}: {e}", "AVISO")
            continue
        tasas[sr] += 1
        if sr / 2 < banda_alta:
            bajos.append((a, sr))
    rep(f"frecuencias de muestreo: {dict(sorted(tasas.items()))}", "OK")
    if bajos:
        rep(f"{len(bajos)} audios con Nyquist < {banda_alta} Hz: la banda alta se rellena "
            f"con el piso, no se estira (ver P3)", "AVISO")

    rep.titulo("Resultado")
    rep(f"{rep.errores} errores, {rep.avisos} avisos")
    print(f"\nReporte: {rep.guardar()}")
    sys.exit(1 if rep.errores else 0)


if __name__ == "__main__":
    main()

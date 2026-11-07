"""
FASE 0 - INVENTARIO Y VERIFICACION DE REDUNDANCIA
=====================================================
Que hace:
  Escanea TODAS las carpetas de anotaciones que existen hoy (no solo
  "Anotaciones/Pulsos", que es lo unico que leian los scripts viejos) y
  responde, de forma reproducible y con evidencia, dos preguntas:

    1. Cuantos archivos/filas hay en cada fuente.
    2. De las fuentes que NO son "Anotaciones/Pulsos" + "Anotaciones/
       Secuencias" (la fuente canonica, organizada por anotador),
       cuantas de sus filas son en realidad EL MISMO pulso/secuencia
       re-exportado con otro nombre de archivo y otra referencia
       temporal, y cuantas son filas genuinamente nuevas que la fuente
       canonica no tiene.

Por que existe esta fase:
  Al analizar los datos se encontro que "Total_Anotaciones" (857 txt) y
  "anotaciones_secuencias_por_audio" (392 txt) NO son anotaciones
  adicionales: son la MISMA anotacion que ya esta en "Anotaciones/
  Pulsos" + "Anotaciones/Secuencias", re-exportada con:
    - Otro nombre de archivo/carpeta (por sitio o por audio, no por
      anotador).
    - Otra referencia temporal (tiempo absoluto de una grabacion
      continua, en vez de tiempo relativo al .wav individual).
    - Codigos de especie abreviados (ej. "SBILI" en vez de
      "saccopteryx bilineata").
  Esto se confirmo comparando (Low Freq, High Freq, Delta Time), que
  son invariantes al re-exportar (no cambian aunque cambie el nombre
  del archivo o el offset de tiempo): el 100% de las filas de esas dos
  fuentes tiene una fila gemela en la fuente canonica.

  Concatenar todo a lo bruto (16.444 archivos) habria triplicado los
  datos sin que ningun script de deduplicacion por nombre de archivo lo
  detectara, porque el nombre de archivo y el tiempo NO coinciden entre
  fuentes (solo Low Freq / High Freq / Delta Time se preservan).

Esta fase no modifica nada: solo audita y deja evidencia en
Fase0_Inventario/. Fase 2 es la que realmente lee y unifica, y SOLO lee
de la fuente canonica (Anotaciones/Pulsos + Anotaciones/Secuencias).

Si en el futuro se agregan anotaciones NUEVAS directamente en
Total_Anotaciones o en anotaciones_secuencias_por_audio (en vez de en
Anotaciones/Pulsos o Secuencias), esta fase las va a detectar como
"filas sin gemela en la fuente canonica" y las va a dejar en un CSV
aparte para revision manual, en vez de perderlas en silencio.
=====================================================
"""

import os
import pandas as pd

import config
from lector_anotaciones import leer_carpeta_anotaciones, columnas_numericas_clave, validar_fuente_canonica


# Subcarpetas de Total_Anotaciones que se auditan como redundantes.
# (Los nombres reales son los sitios/especies del proyecto; si cambian
# de un año a otro basta con actualizar esta lista.)
SUBCARPETAS_TOTAL_POR_SITIO = [
    "Molossus I", "Molossus Rufus", "Molossus molossus",
    "IALL", "PEdificios", "Enero2024",
]


def clave_contenido(df):
    """
    Clave de comparacion ENTRE FUENTES DISTINTAS. No se puede usar
    Begin File + Begin Time (cambian entre re-exportaciones), asi que
    se usa lo que SI es invariante: el par de frecuencias y la
    duracion del evento, redondeados para tolerar ruido de precision
    de punto flotante entre exportaciones.
    """
    df = columnas_numericas_clave(df)
    return list(zip(
        df["Low Freq (Hz)"].round(3),
        df["High Freq (Hz)"].round(3),
        df["Delta Time (s)"].round(4),
    ))


def auditar_fuente_redundante(nombre_fuente, ruta_fuente, claves_canonicas, reporte):
    df, errores = leer_carpeta_anotaciones(ruta_fuente, etiqueta_grupo=nombre_fuente)

    if df.empty:
        print(f"[{nombre_fuente}] no se encontraron .txt en {ruta_fuente} (se omite)")
        return

    df["__clave"] = clave_contenido(df)
    coincide = df["__clave"].isin(claves_canonicas)

    n_total = len(df)
    n_coincide = int(coincide.sum())
    n_nuevas = n_total - n_coincide
    pct = 100 * n_coincide / n_total if n_total else 0

    print(f"\n[{nombre_fuente}]")
    print(f"  Archivos .txt        : {df['Ruta_TXT'].nunique()}")
    print(f"  Filas totales        : {n_total}")
    print(f"  Con gemela en la fuente canonica (redundante): {n_coincide} ({pct:.1f}%)")
    print(f"  SIN gemela (posible dato nuevo)              : {n_nuevas}")

    reporte.append({
        "fuente": nombre_fuente,
        "ruta": ruta_fuente,
        "archivos_txt": df["Ruta_TXT"].nunique(),
        "filas_totales": n_total,
        "filas_redundantes": n_coincide,
        "filas_sin_gemela": n_nuevas,
        "pct_redundante": round(pct, 2),
    })

    if n_nuevas > 0:
        ruta_salida = os.path.join(
            config.SALIDA_FASE0, f"filas_nuevas_en_{nombre_fuente}.csv"
        )
        df.loc[~coincide].drop(columns="__clave").to_csv(ruta_salida, index=False)
        print(f"  -> {n_nuevas} filas sin gemela guardadas para revision manual en:")
        print(f"     {ruta_salida}")
        print(
            "     ACCION REQUERIDA: revisar si son anotaciones genuinamente "
            "nuevas (en ese caso, deben incorporarse a Anotaciones/Pulsos o "
            "Anotaciones/Secuencias, la fuente canonica, para que fase2 las "
            "tome en cuenta) o si son variaciones de precision numerica."
        )


def main():
    print("=" * 60)
    print("FASE 0 - INVENTARIO Y VERIFICACION DE REDUNDANCIA")
    print("=" * 60)

    # -------------------------------------------------
    # 1. Fuente canonica
    # -------------------------------------------------
    df_pulsos, err_p = leer_carpeta_anotaciones(config.RUTA_PULSOS, "pulsos_por_anotador")
    df_secuencias, err_s = leer_carpeta_anotaciones(config.RUTA_SECUENCIAS, "secuencias_por_anotador")

    print("\n[FUENTE CANONICA] Anotaciones/Pulsos + Anotaciones/Secuencias")
    print(f"  Pulsos.txt      : {df_pulsos['Ruta_TXT'].nunique() if not df_pulsos.empty else 0} archivos, {len(df_pulsos)} filas")
    print(f"  Secuencias.txt  : {df_secuencias['Ruta_TXT'].nunique() if not df_secuencias.empty else 0} archivos, {len(df_secuencias)} filas")

    if not validar_fuente_canonica(df_pulsos, df_secuencias, config):
        return
    config.crear_carpetas_salida()

    canonica = pd.concat([df_pulsos, df_secuencias], ignore_index=True)
    claves_canonicas = set(clave_contenido(canonica))

    duplicados_internos = canonica[
        columnas_numericas_clave(canonica).duplicated(
            subset=["Begin File", "Begin Time (s)", "End Time (s)", "Low Freq (Hz)", "High Freq (Hz)"],
            keep=False,
        )
    ]
    print(f"  Duplicados internos en la fuente canonica: {len(duplicados_internos)}")

    if err_p or err_s:
        print(f"  [AVISO] errores de lectura: {len(err_p) + len(err_s)} (ver reporte)")

    # -------------------------------------------------
    # 2. Fuentes a auditar como redundantes
    # -------------------------------------------------
    reporte = []

    auditar_fuente_redundante(
        "total_anotaciones_pulsos",
        os.path.join(config.RUTA_TOTAL_ANOTACIONES, "Pulsos"),
        claves_canonicas, reporte,
    )
    auditar_fuente_redundante(
        "total_anotaciones_secuencias",
        os.path.join(config.RUTA_TOTAL_ANOTACIONES, "Secuencias"),
        claves_canonicas, reporte,
    )
    for sub in SUBCARPETAS_TOTAL_POR_SITIO:
        ruta_sub = os.path.join(config.RUTA_TOTAL_ANOTACIONES, sub)
        if os.path.isdir(ruta_sub):
            auditar_fuente_redundante(
                f"total_anotaciones_sitio_{sub.replace(' ', '_')}",
                ruta_sub, claves_canonicas, reporte,
            )
    auditar_fuente_redundante(
        "secuencias_por_audio_saccopteryx",
        config.RUTA_SECUENCIAS_POR_AUDIO,
        claves_canonicas, reporte,
    )

    # -------------------------------------------------
    # 3. Fuera de alcance (solo se deja documentado)
    # -------------------------------------------------
    print(f"\n[FUERA DE ALCANCE] {config.RUTA_SALIDAS_MODELO_DETECCION}")
    print(
        "  Son salidas de clasificacion automatica de un modelo "
        "(FasterRCNN), no anotacion manual. No se procesan en este "
        "pipeline de anotaciones."
    )

    # -------------------------------------------------
    # 4. Guardar reporte
    # -------------------------------------------------
    ruta_reporte = os.path.join(config.SALIDA_FASE0, "reporte_inventario_fuentes.csv")
    pd.DataFrame(reporte).to_csv(ruta_reporte, index=False)

    print("\n" + "=" * 60)
    print("FASE 0 FINALIZADA")
    print("=" * 60)
    print(f"Reporte de inventario: {ruta_reporte}")
    print(
        "\nSi TODAS las fuentes no-canonicas salieron en ~100% redundante "
        "y sin archivos 'filas_nuevas_en_*.csv', fase2 puede correr solo "
        "sobre Anotaciones/Pulsos + Anotaciones/Secuencias con confianza "
        "de que no se esta perdiendo ningun dato real."
    )


if __name__ == "__main__":
    main()

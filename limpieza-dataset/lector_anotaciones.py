"""
lector_anotaciones.py
=====================================================
Utilidades compartidas para leer los .txt de anotaciones (tablas de
seleccion de Raven Pro, separadas por tabulador).

Antes cada script (normalizar_anotaciones.py, limpiar_anotaciones_pulsos.py,
audit_anotacion_especie.py, audit_anotacion_calidad.py,
audit_anotacion_pulsetype.py...) repetia el mismo bucle os.walk +
pd.read_csv con pequenas variaciones. Se centraliza aqui una sola vez.
=====================================================
"""

import os
import pandas as pd


def leer_carpeta_anotaciones(ruta_raiz, etiqueta_grupo=None):
    """
    Recorre ruta_raiz recursivamente, lee todos los .txt como tablas de
    seleccion de Raven (separadas por tab) y los concatena en un solo
    DataFrame.

    Agrega columnas de trazabilidad:
      - Archivo_TXT   : nombre del archivo
      - Ruta_TXT      : ruta completa
      - Grupo_Fuente  : etiqueta_grupo (si se paso), util para saber de
                        que carpeta/fuente vino cada fila despues de
                        concatenar varias fuentes.

    Devuelve (df, errores) donde errores es una lista de dicts
    {"Archivo": ..., "Error": ...} para los txt que no se pudieron leer
    (nunca se descartan en silencio).
    """
    registros = []
    errores = []

    if not os.path.isdir(ruta_raiz):
        return pd.DataFrame(), [{"Archivo": ruta_raiz, "Error": "La carpeta no existe"}]

    for root, _, archivos in os.walk(ruta_raiz):
        for archivo in archivos:
            if not archivo.lower().endswith(".txt"):
                continue

            ruta_txt = os.path.join(root, archivo)

            try:
                df = pd.read_csv(ruta_txt, sep="\t", encoding="utf-8", dtype=str)
                df["Archivo_TXT"] = archivo
                df["Ruta_TXT"] = ruta_txt
                if etiqueta_grupo is not None:
                    df["Grupo_Fuente"] = etiqueta_grupo
                registros.append(df)
            except Exception as e:
                errores.append({"Archivo": ruta_txt, "Error": str(e)})

    if not registros:
        return pd.DataFrame(), errores

    df_total = pd.concat(registros, ignore_index=True)
    return df_total, errores


def columnas_numericas_clave(df):
    """
    Convierte a numerico (float) las columnas usadas como clave de
    duplicados / comparacion entre fuentes. No falla si faltan columnas
    (devuelve el df igual, solo convierte las que existan).
    """
    df = df.copy()
    for col in ["Low Freq (Hz)", "High Freq (Hz)", "Delta Time (s)",
                "Begin Time (s)", "End Time (s)"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def validar_fuente_canonica(df_pulsos, df_secuencias, config):
    """
    Chequeo comun a fase0/fase1/fase2: si no se encontro NINGUN .txt en
    Anotaciones/Pulsos ni Anotaciones/Secuencias, seguir ejecutando
    produce errores confusos mas adelante (columnas que no existen en
    un DataFrame vacio). En vez de eso, se imprime un diagnostico claro
    de rutas y se devuelve False para que el script se detenga ahi.
    Devuelve True si hay datos y esta bien continuar.
    """
    if not df_pulsos.empty or not df_secuencias.empty:
        return True

    print("\n[ERROR] No se encontro ningun .txt en la fuente canonica. Antes de seguir, revisa:")
    print(f"  config.RUTA_BASE       = {config.RUTA_BASE}")
    print(f"  config.RUTA_PULSOS     = {config.RUTA_PULSOS}")
    print(f"  config.RUTA_SECUENCIAS = {config.RUTA_SECUENCIAS}")
    print(f"  ¿existe RUTA_BASE?       {os.path.isdir(config.RUTA_BASE)}")
    print(f"  ¿existe RUTA_PULSOS?     {os.path.isdir(config.RUTA_PULSOS)}")
    print(f"  ¿existe RUTA_SECUENCIAS? {os.path.isdir(config.RUTA_SECUENCIAS)}")
    print("\nSi las rutas de arriba no son correctas para esta maquina, corrige RUTA_BASE en")
    print("config.py, o exporta ANOTACIONES_BASE=/ruta/correcta antes de correr el script.")
    return False

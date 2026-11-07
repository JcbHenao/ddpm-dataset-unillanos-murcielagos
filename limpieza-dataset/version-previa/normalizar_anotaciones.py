import os
import pandas as pd
from collections import Counter


# =====================================================
# Rutas
# =====================================================

RUTA_ANOTACIONES = (
    "/media/adalab/bodega/CamiloBuitrago/Anotaciones/Pulsos"
)

RUTA_MAPEO_ESPECIES = (
    "/media/adalab/bodega/CamiloBuitrago/"
    "Generacion_Espectrogramas/mapeo_especies.csv"
)

RUTA_MAPEO_CALIDAD = (
    "/media/adalab/bodega/CamiloBuitrago/"
    "Generacion_Espectrogramas/mapeo_calidad.csv"
)

RUTA_MAPEO_PULSE_TYPE = (
    "/media/adalab/bodega/CamiloBuitrago/"
    "Generacion_Espectrogramas/mapeo_pulse_type.csv"
)

SALIDA = (
    "/media/adalab/bodega/CamiloBuitrago/"
    "Generacion_Espectrogramas/Limpieza_Anotaciones_Pulsos"
)

os.makedirs(SALIDA, exist_ok=True)

SALIDA_ESPECIES = os.path.join(SALIDA, "Por_Especie")
os.makedirs(SALIDA_ESPECIES, exist_ok=True)


# =====================================================
# Fusion final de Calidad: de 5 categorias a 3
# =====================================================
# El diccionario mapeo_calidad.csv normaliza typos a 5 categorias:
# Baja, Alta, Buena, Regular, Media.
# Aqui se funden esas 5 en las 3 categorias finales de uso:
#   Buena   -> Alta
#   Regular -> Media
#   Baja, Alta, Media se quedan igual

FUSION_CALIDAD_FINAL = {
    "Baja": "Baja",
    "Alta": "Alta",
    "Buena": "Alta",
    "Regular": "Media",
    "Media": "Media",
}


# =====================================================
# Cargar mapeos
# =====================================================

def cargar_mapeo(ruta, columna_original, columna_final):
    df = pd.read_csv(ruta, dtype=str)
    return dict(
        zip(
            df[columna_original].str.strip(),
            df[columna_final].str.strip(),
        )
    )


mapeo_especies = cargar_mapeo(
    RUTA_MAPEO_ESPECIES,
    "Especie_original",
    "Especie",
)

mapeo_calidad = cargar_mapeo(
    RUTA_MAPEO_CALIDAD,
    "Calidad_original",
    "Calidad",
)

mapeo_pulse_type = cargar_mapeo(
    RUTA_MAPEO_PULSE_TYPE,
    "Pulse_Type_original",
    "Pulse_Type",
)


# =====================================================
# Leer TXT
# =====================================================

registros = []
errores = []
contador_txt = 0

for root, _, archivos in os.walk(RUTA_ANOTACIONES):
    for archivo in archivos:
        if not archivo.lower().endswith(".txt"):
            continue

        ruta_txt = os.path.join(root, archivo)

        try:
            df = pd.read_csv(ruta_txt, sep="\t", encoding="utf-8")
            df["Archivo_TXT"] = archivo
            df["Ruta_TXT"] = ruta_txt
            registros.append(df)
            contador_txt += 1
        except Exception as e:
            errores.append({"Archivo": archivo, "Error": str(e)})

print("TXT procesados:", contador_txt)

if len(registros) == 0:
    raise Exception("No se encontraron anotaciones")

df = pd.concat(registros, ignore_index=True)
print("Registros originales:", len(df))


# =====================================================
# Reubicacion de Molcfruf: valor de Specie que se filtro por error
# a la columna Pulse Type
# =====================================================
# 'Molcfruf' es una contraccion de 'MolossusRufus' que aparecio por
# error de captura en la columna Pulse Type en vez de Specie. Estas 9
# filas se corrigen ANTES de la normalizacion de especie: se sobre-
# escribe Specie_original con el valor correcto, y se vacia Pulse Type
# (el valor real de nivel/duracion para esas filas se perdio).

mask_molcfruf = df["Pulse Type"].astype(str).str.strip() == "Molcfruf"
filas_molcfruf_corregidas = int(mask_molcfruf.sum())

df.loc[mask_molcfruf, "Specie"] = "MolossusRufus"
df.loc[mask_molcfruf, "Pulse Type"] = pd.NA


# =====================================================
# Normalizacion especie
# =====================================================

especies_originales = []
for idx, fila in df.iterrows():
    original = str(fila.get("Specie", "")).strip()
    especies_originales.append(original)

df["Specie_original"] = especies_originales
df["Specie"] = df["Specie_original"].map(mapeo_especies)

# eliminar desconocidos
mask_especie = df["Specie"].notna()
descartados_especie = df[~mask_especie].copy()
df = df[mask_especie].copy()


# =====================================================
# Normalizacion calidad (typos -> 5 categorias)
# =====================================================

df["Calidad_original"] = df["Calidad"].astype(str).str.strip()

df["Calidad"] = (
    df["Calidad_original"]
    .map(mapeo_calidad)
    .fillna(df["Calidad_original"])
)


# =====================================================
# Fusion final de Calidad (5 -> 3 categorias)
# =====================================================
# Se aplica DESPUES de la normalizacion de typos. Los valores que no
# esten en FUSION_CALIDAD_FINAL (ej. typos no contemplados en el
# diccionario original) se dejan tal cual, y quedan registrados en
# 'calidad_no_fusionada.csv' para revision manual, en vez de perderse
# silenciosamente.

df["Calidad_5_categorias"] = df["Calidad"]

df["Calidad"] = df["Calidad_5_categorias"].map(FUSION_CALIDAD_FINAL)

mask_calidad_no_fusionada = df["Calidad"].isna() & df["Calidad_5_categorias"].notna()

# si no se pudo fusionar, conservar el valor de las 5 categorias en vez
# de dejar la fila en NA, para no perder informacion silenciosamente
df.loc[mask_calidad_no_fusionada, "Calidad"] = df.loc[
    mask_calidad_no_fusionada, "Calidad_5_categorias"
]

# copiar DESPUES del fillna, para que el CSV de auditoria muestre el
# valor conservado real en la columna Calidad, no NaN
calidad_no_fusionada = df[mask_calidad_no_fusionada].copy()

# =====================================================
# Normalizacion Pulse Type (typos -> 5 categorias: Bajo/Medio/Alto/corto/largo)
# =====================================================
# Pulse Type es una variable propia e independiente de Calidad (varia
# de forma independiente en los datos reales, confirmado por auditoria).
# Se normaliza con su PROPIO diccionario (mapeo_pulse_type.csv), nunca
# con mapeo_calidad, aunque algunas palabras (Bajo/Medio/Alto) coincidan
# lexicamente con sinonimos de Calidad -- son variables distintas que
# comparten vocabulario por coincidencia, no por error de captura.
#
# Las filas con 'Molcfruf' ya se reubicaron antes de este punto (su
# Pulse Type ya quedo en NA), por lo que no pasan por este mapeo.

df["Pulse_Type_original"] = df["Pulse Type"].astype(str).str.strip()

# evitar mapear la cadena "nan" generada por valores NA al convertir a str
df.loc[df["Pulse Type"].isna(), "Pulse_Type_original"] = pd.NA

df["Pulse Type"] = df["Pulse_Type_original"].map(mapeo_pulse_type)

mask_pulse_type_no_mapeado = (
    df["Pulse Type"].isna() & df["Pulse_Type_original"].notna()
)

# si no se pudo mapear (typo nuevo no contemplado), conservar el valor
# original en vez de perderlo, igual que se hace con Calidad
df.loc[mask_pulse_type_no_mapeado, "Pulse Type"] = df.loc[
    mask_pulse_type_no_mapeado, "Pulse_Type_original"
]

# copiar DESPUES del fillna, para que el CSV de auditoria muestre el
# valor conservado real, no NaN
pulse_type_no_mapeado = df[mask_pulse_type_no_mapeado].copy()


# =====================================================
# Eliminacion duplicados
# =====================================================

columnas_id = [
    "Begin File",
    "Begin Time (s)",
    "End Time (s)",
    "Low Freq (Hz)",
    "High Freq (Hz)",
]

duplicados = df[df.duplicated(subset=columnas_id, keep=False)].copy()
df = df.drop_duplicates(subset=columnas_id, keep="first")


# =====================================================
# Crear ID unico
# =====================================================

contador_especie = Counter()
ids = []

for especie in df["Specie"]:
    contador_especie[especie] += 1
    ids.append(f"{especie}_{contador_especie[especie]:06d}")

df["Pulse_ID"] = ids


# =====================================================
# Guardar maestro
# =====================================================

archivo_final = os.path.join(SALIDA, "anotaciones_pulsos_limpias.csv")
df.to_csv(archivo_final, index=False, encoding="utf-8")


# =====================================================
# Separar por especie
# =====================================================

for especie, grupo in df.groupby("Specie"):
    nombre = especie + ".csv"
    grupo.to_csv(
        os.path.join(SALIDA_ESPECIES, nombre),
        index=False,
        encoding="utf-8",
    )


# =====================================================
# Auditorias
# =====================================================

duplicados.to_csv(os.path.join(SALIDA, "duplicados.csv"), index=False)

pd.DataFrame(descartados_especie).to_csv(
    os.path.join(SALIDA, "descartados_especie.csv"), index=False
)

if len(calidad_no_fusionada):
    calidad_no_fusionada.to_csv(
        os.path.join(SALIDA, "calidad_no_fusionada.csv"), index=False
    )

if len(pulse_type_no_mapeado):
    pulse_type_no_mapeado.to_csv(
        os.path.join(SALIDA, "pulse_type_no_mapeado.csv"), index=False
    )

if len(errores):
    pd.DataFrame(errores).to_csv(
        os.path.join(SALIDA, "errores_txt.csv"), index=False
    )

resumen = df["Specie"].value_counts().reset_index()
resumen.columns = ["Especie", "Cantidad"]
resumen.to_csv(os.path.join(SALIDA, "resumen_especies.csv"), index=False)

resumen_calidad = df["Calidad"].value_counts().reset_index()
resumen_calidad.columns = ["Calidad", "Cantidad"]
resumen_calidad.to_csv(
    os.path.join(SALIDA, "resumen_calidad.csv"), index=False
)

resumen_pulse_type = df["Pulse Type"].value_counts(dropna=False).reset_index()
resumen_pulse_type.columns = ["Pulse Type", "Cantidad"]
resumen_pulse_type.to_csv(
    os.path.join(SALIDA, "resumen_pulse_type.csv"), index=False
)


# =====================================================
# Reporte
# =====================================================

print("\n==============================")
print("NORMALIZACION FINALIZADA")
print("==============================")
print("Registros finales:", len(df))
print("Duplicados eliminados:", len(duplicados))
print("Descartados especie:", len(descartados_especie))
print("Calidad no fusionada (revisar):", len(calidad_no_fusionada))
print("Filas Molcfruf reubicadas a Specie=MolossusRufus:", filas_molcfruf_corregidas)
print("Pulse Type no mapeado (revisar):", len(pulse_type_no_mapeado))

print("\nDistribucion especies:")
print(resumen.to_string(index=False))

print("\nDistribucion calidad (3 categorias finales):")
print(resumen_calidad.to_string(index=False))

print("\nDistribucion Pulse Type (5 categorias finales):")
print(resumen_pulse_type.to_string(index=False))

print("\nArchivo:", archivo_final)

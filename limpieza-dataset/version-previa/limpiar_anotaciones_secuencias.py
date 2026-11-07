import os
import pandas as pd
from collections import Counter


# =====================================================
# Rutas
# =====================================================

RUTA_ANOTACIONES = (
    "/media/adalab/bodega/CamiloBuitrago/Total_Anotaciones"
)


RUTA_MAPEO_ESPECIES = (
    "/media/adalab/bodega/CamiloBuitrago/"
    "Generacion_Espectrogramas/mapeo_especies.csv"
)


RUTA_MAPEO_CALIDAD = (
    "/media/adalab/bodega/CamiloBuitrago/"
    "Generacion_Espectrogramas/mapeo_calidad.csv"
)


SALIDA = (
    "/media/adalab/bodega/CamiloBuitrago/"
    "Generacion_Espectrogramas/Limpieza_Anotaciones"
)



os.makedirs(
    SALIDA,
    exist_ok=True
)


SALIDA_ESPECIES = os.path.join(
    SALIDA,
    "Por_Especie"
)


os.makedirs(
    SALIDA_ESPECIES,
    exist_ok=True
)



# =====================================================
# Cargar mapeos
# =====================================================

def cargar_mapeo(ruta, columna_original, columna_final):

    df = pd.read_csv(
        ruta,
        dtype=str
    )

    return dict(
        zip(
            df[columna_original].str.strip(),
            df[columna_final].str.strip()
        )
    )



mapeo_especies = cargar_mapeo(
    RUTA_MAPEO_ESPECIES,
    "Especie_original",
    "Especie"
)


mapeo_calidad = cargar_mapeo(
    RUTA_MAPEO_CALIDAD,
    "Calidad_original",
    "Calidad"
)



# =====================================================
# Leer TXT
# =====================================================

registros = []

errores = []



contador_txt = 0



for root, _, archivos in os.walk(
    RUTA_ANOTACIONES
):

    for archivo in archivos:


        if not archivo.lower().endswith(".txt"):
            continue


        ruta_txt = os.path.join(
            root,
            archivo
        )


        try:

            df = pd.read_csv(
                ruta_txt,
                sep="\t",
                encoding="utf-8"
            )


            df["Archivo_TXT"] = archivo


            df["Ruta_TXT"] = ruta_txt


            registros.append(
                df
            )


            contador_txt += 1



        except Exception as e:


            errores.append(
                {
                    "Archivo": archivo,
                    "Error": str(e)
                }
            )



print(
    "TXT procesados:",
    contador_txt
)



if len(registros) == 0:

    raise Exception(
        "No se encontraron anotaciones"
    )



df = pd.concat(
    registros,
    ignore_index=True
)



print(
    "Registros originales:",
    len(df)
)



# =====================================================
# Normalizacion especie
# =====================================================

descartados_especie = []


especies_originales = []



for idx, fila in df.iterrows():


    original = str(
        fila.get(
            "Specie",
            ""
        )
    ).strip()


    especies_originales.append(
        original
    )



df["Specie_original"] = especies_originales



df["Specie"] = (
    df["Specie_original"]
    .map(
        mapeo_especies
    )
)



# eliminar desconocidos

mask_especie = (
    df["Specie"].notna()
)



descartados_especie = df[
    ~mask_especie
].copy()



df = df[
    mask_especie
].copy()



# =====================================================
# Normalizacion calidad
# =====================================================


df["Calidad_original"] = (
    df["Calidad"]
    .astype(str)
    .str.strip()
)


df["Calidad"] = (
    df["Calidad_original"]
    .map(
        mapeo_calidad
    )
    .fillna(
        df["Calidad_original"]
    )
)



# =====================================================
# Eliminacion duplicados
# =====================================================

columnas_id = [

    "Begin File",

    "Begin Time (s)",

    "End Time (s)",

    "Low Freq (Hz)",

    "High Freq (Hz)"

]



duplicados = df[
    df.duplicated(
        subset=columnas_id,
        keep=False
    )
].copy()



df = df.drop_duplicates(
    subset=columnas_id,
    keep="first"
)



# =====================================================
# Crear ID unico
# =====================================================

contador_especie = Counter()



ids = []



for especie in df["Specie"]:

    contador_especie[especie] += 1

    ids.append(
        f"{especie}_{contador_especie[especie]:06d}"
    )



df["Pulse_ID"] = ids



# =====================================================
# Guardar maestro
# =====================================================

archivo_final = os.path.join(
    SALIDA,
    "anotaciones_pulsos_limpias.csv"
)



df.to_csv(
    archivo_final,
    index=False,
    encoding="utf-8"
)



# =====================================================
# Separar por especie
# =====================================================


for especie, grupo in df.groupby(
    "Specie"
):


    nombre = (
        especie
        +
        ".csv"
    )


    grupo.to_csv(

        os.path.join(
            SALIDA_ESPECIES,
            nombre
        ),

        index=False,

        encoding="utf-8"

    )



# =====================================================
# Auditorias
# =====================================================


duplicados.to_csv(

    os.path.join(
        SALIDA,
        "duplicados.csv"
    ),

    index=False

)



pd.DataFrame(
    descartados_especie
).to_csv(

    os.path.join(
        SALIDA,
        "descartados_especie.csv"
    ),

    index=False

)



if len(errores):

    pd.DataFrame(
        errores
    ).to_csv(

        os.path.join(
            SALIDA,
            "errores_txt.csv"
        ),

        index=False

    )



resumen = (
    df["Specie"]
    .value_counts()
    .reset_index()
)



resumen.columns = [
    "Especie",
    "Cantidad"
]


resumen.to_csv(

    os.path.join(
        SALIDA,
        "resumen_especies.csv"
    ),

    index=False

)



# =====================================================
# Reporte
# =====================================================

print("\n==============================")
print("NORMALIZACION FINALIZADA")
print("==============================")


print(
    "Registros finales:",
    len(df)
)


print(
    "Duplicados eliminados:",
    len(duplicados)
)


print(
    "Descartados especie:",
    len(descartados_especie)
)


print("\nDistribucion:")

print(
    resumen.to_string(
        index=False
    )
)



print(
    "\nArchivo:",
    archivo_final
)
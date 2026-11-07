import os
import pandas as pd
from collections import Counter


# -------------------------------------------------
# Configuracion
# -------------------------------------------------

RUTA_ANOTACIONES = (
    "/media/adalab/bodega/CamiloBuitrago/Anotaciones/Pulsos"
)


SALIDA_CSV = (
    "/media/adalab/bodega/CamiloBuitrago/"
    "resumen_calidad_originales.csv"
)


# -------------------------------------------------
# Variables
# -------------------------------------------------

contador_calidad = Counter()

archivos_procesados = 0
archivos_error = 0
filas_totales = 0


# -------------------------------------------------
# Procesamiento
# -------------------------------------------------

for root, _, files in os.walk(RUTA_ANOTACIONES):

    for archivo in files:

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


        except Exception as e:

            print(
                f"Error leyendo {ruta_txt}: {e}"
            )

            archivos_error += 1
            continue



        archivos_procesados += 1


        # Validar columna Calidad

        if "Calidad" not in df.columns:

            print(
                f"Sin columna Calidad: {ruta_txt}"
            )

            continue



        filas_totales += len(df)



        calidades = (
            df["Calidad"]
            .dropna()
            .astype(str)
            .str.strip()
        )


        for calidad in calidades:

            contador_calidad[calidad] += 1



# -------------------------------------------------
# Resultados
# -------------------------------------------------

resultado = pd.DataFrame(
    contador_calidad.items(),
    columns=[
        "Calidad_original",
        "Cantidad"
    ]
)


resultado = resultado.sort_values(
    by="Cantidad",
    ascending=False
)


resultado.to_csv(
    SALIDA_CSV,
    index=False,
    encoding="utf-8"
)


# -------------------------------------------------
# Reporte consola
# -------------------------------------------------

print("\n==============================")
print("AUDITORIA CALIDAD FINALIZADA")
print("==============================")


print(
    f"Archivos TXT procesados: {archivos_procesados}"
)

print(
    f"Archivos con error: {archivos_error}"
)

print(
    f"Total filas anotadas: {filas_totales}"
)


print(
    f"\nValores de Calidad encontrados: {len(resultado)}"
)


print("\nListado:")

for _, fila in resultado.iterrows():

    print(
        f"{fila['Calidad_original']:20s}"
        f" -> {fila['Cantidad']}"
    )


print(
    "\nCSV generado:"
)

print(
    SALIDA_CSV
)

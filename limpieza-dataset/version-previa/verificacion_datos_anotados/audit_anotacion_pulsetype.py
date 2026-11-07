import os
import pandas as pd


# =====================================================
# Rutas
# =====================================================

RUTA_ANOTACIONES = (
    "/media/adalab/bodega/CamiloBuitrago/Anotaciones/Pulsos"
)

RUTA_MAPEO_CALIDAD = (
    "/media/adalab/bodega/CamiloBuitrago/"
    "Generacion_Espectrogramas/mapeo_calidad.csv"
)

RUTA_SALIDA = (
    "/media/adalab/bodega/CamiloBuitrago/"
    "resumen_pulse_type_crudo.csv"
)


# =====================================================
# Leer TXT crudos
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
            registros.append(df)
            contador_txt += 1
        except Exception as e:
            errores.append({"Archivo": archivo, "Error": str(e)})

print("==============================")
print("AUDITORIA PULSE TYPE (datos crudos)")
print("==============================")
print("Archivos TXT procesados:", contador_txt)
print("Archivos con error:", len(errores))

if len(registros) == 0:
    raise Exception("No se encontraron anotaciones")

df = pd.concat(registros, ignore_index=True)
print("Total filas anotadas:", len(df))


# =====================================================
# Listado completo de valores de Pulse Type, sin agrupar
# =====================================================
# A diferencia de la auditoria de similitud, aqui se listan TODAS las
# formas exactas tal como aparecen en el .txt, sin sugerir agrupaciones,
# para inspeccion manual completa antes de decidir cualquier normalizacion.

conteo_pulse_type = (
    df["Pulse Type"]
    .astype(str)
    .str.strip()
    .value_counts()
)

print()
print("Valores de Pulse Type encontrados (formas exactas):", len(conteo_pulse_type))
print()
print("Listado completo:")
for valor, cantidad in conteo_pulse_type.items():
    print(f"{valor!r:25} -> {cantidad}")


# =====================================================
# Verificacion cruzada con el diccionario de Calidad
# =====================================================
# Confirma si el mapeo_calidad.csv se esta aplicando (o podria aplicarse
# por error) sobre valores de Pulse Type. Si alguna forma de Pulse Type
# coincide EXACTAMENTE con una clave de Calidad_original, se reporta
# como alerta para revisar manualmente, sin modificar nada.

print()
print("------------------------------")
print("VERIFICACION CRUCE CON mapeo_calidad.csv")
print("------------------------------")

try:
    mapeo_calidad_df = pd.read_csv(RUTA_MAPEO_CALIDAD, dtype=str)
    claves_calidad = set(
        mapeo_calidad_df["Calidad_original"].str.strip()
    )

    valores_pulse_type = set(conteo_pulse_type.index)

    interseccion = valores_pulse_type & claves_calidad

    if interseccion:
        print(
            f"ALERTA: {len(interseccion)} valores de Pulse Type "
            f"coinciden exactamente con claves de mapeo_calidad.csv:"
        )
        for valor in sorted(interseccion):
            print(
                f"  '{valor}' -> "
                f"{conteo_pulse_type[valor]} filas en Pulse Type"
            )
        print(
            "\nIMPORTANTE: esto NO significa que el mapeo se haya "
            "aplicado sobre Pulse Type -- solo indica que, SI se "
            "aplicara por error, estos valores cambiarian. Verificar "
            "manualmente en el script de normalizacion que "
            "df['Pulse Type'] nunca pase por mapeo_calidad."
        )
    else:
        print(
            "Sin coincidencias exactas entre los valores de Pulse Type "
            "y las claves de mapeo_calidad.csv."
        )

except FileNotFoundError:
    print(
        f"[AVISO] No se encontro {RUTA_MAPEO_CALIDAD}, "
        "se omite la verificacion cruzada."
    )


# =====================================================
# Guardar CSV
# =====================================================

resumen = conteo_pulse_type.reset_index()
resumen.columns = ["Pulse Type (forma exacta)", "Cantidad"]
resumen.to_csv(RUTA_SALIDA, index=False)

print()
print("CSV generado:")
print(RUTA_SALIDA)

print()
print("==============================")
print("AUDITORIA FINALIZADA")
print("==============================")
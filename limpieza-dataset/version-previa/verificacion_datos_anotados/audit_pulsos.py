import pandas as pd


# =====================================================
# Ruta
# =====================================================

RUTA = (
    "/media/adalab/bodega/CamiloBuitrago/Generacion_Espectrogramas/Limpieza_Anotaciones_Pulsos/anotaciones_pulsos_limpias.csv"
)



# =====================================================
# Cargar
# =====================================================

df = pd.read_csv(RUTA)


print("==============================")
print("AUDITORIA PULSOS")
print("==============================")

print(
    "Total pulsos:",
    len(df)
)



# =====================================================
# Duracion temporal
# =====================================================

print("\nDURACION PULSOS")
print("------------------------------")


for nombre, valor in {

    "Promedio":
    df["Delta Time (s)"].mean(),

    "Mediana":
    df["Delta Time (s)"].median(),

    "Minimo":
    df["Delta Time (s)"].min(),

    "Maximo":
    df["Delta Time (s)"].max(),

    "Percentil 95":
    df["Delta Time (s)"].quantile(0.95),

    "Percentil 99":
    df["Delta Time (s)"].quantile(0.99)

}.items():

    print(
        f"{nombre:15}: {valor:.6f} s"
    )



# =====================================================
# Frecuencias
# =====================================================

print("\nFRECUENCIAS")
print("------------------------------")


print(
    "Low Freq minimo:",
    df["Low Freq (Hz)"].min(),
    "Hz"
)


print(
    "Low Freq promedio:",
    df["Low Freq (Hz)"].mean(),
    "Hz"
)


print(
    "Peak Freq minimo:",
    df["Peak Freq (Hz)"].min(),
    "Hz"
)


print(
    "Peak Freq promedio:",
    df["Peak Freq (Hz)"].mean(),
    "Hz"
)


print(
    "Peak Freq maximo:",
    df["Peak Freq (Hz)"].max(),
    "Hz"
)


print(
    "High Freq maximo:",
    df["High Freq (Hz)"].max(),
    "Hz"
)



# =====================================================
# Analisis por especie
# =====================================================

print("\nDURACION POR ESPECIE")
print("------------------------------")


tabla_especies = (
    df
    .groupby("Specie")
    ["Delta Time (s)"]
    .agg(
        cantidad="count",
        promedio="mean",
        mediana="median",
        minimo="min",
        maximo="max",
        p95=lambda x: x.quantile(0.95)
    )
    .sort_values(
        "cantidad",
        ascending=False
    )
)


print(tabla_especies)



# =====================================================
# Tipos de pulso
# =====================================================

if "Pulse Type" in df.columns:

    print("\nTIPOS DE PULSO")
    print("------------------------------")

    print(
        df["Pulse Type"]
        .value_counts()
    )



# =====================================================
# Calidad
# =====================================================

if "Calidad" in df.columns:

    print("\nCALIDAD")
    print("------------------------------")

    print(
        df["Calidad"]
        .value_counts()
    )



# =====================================================
# Señal
# =====================================================

if "Signal Type" in df.columns:

    print("\nTIPO DE SEÑAL")
    print("------------------------------")

    print(
        df["Signal Type"]
        .value_counts()
    )



print("\n==============================")
print("AUDITORIA FINALIZADA")
print("==============================")
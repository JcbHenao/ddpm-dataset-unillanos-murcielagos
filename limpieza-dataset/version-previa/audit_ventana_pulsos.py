import pandas as pd
import numpy as np


# =====================================================
# Ruta
# =====================================================

RUTA_ANOTACIONES_LIMPIAS = (
    "/media/adalab/bodega/CamiloBuitrago/"
    "Generacion_Espectrogramas/Limpieza_Anotaciones_Pulsos/"
    "anotaciones_pulsos_limpias.csv"
)

# Debe coincidir con DURACION_OBJETIVO en generar_espectrogramas.py
DURACION_OBJETIVO = 0.015  # 15 ms


# =====================================================
# Cargar
# =====================================================

df = pd.read_csv(RUTA_ANOTACIONES_LIMPIAS)

print("==============================")
print("AUDITORIA: VENTANA FIJA vs DURACION REAL DEL PULSO")
print("==============================")
print("Total filas:", len(df))
print(f"Ventana fija actual (DURACION_OBJETIVO): {DURACION_OBJETIVO*1000:.1f} ms")
print()


# =====================================================
# Duracion real de cada pulso
# =====================================================
# Delta Time (s) ya viene calculado en el TXT original (End - Begin).
# Si por algun motivo no esta disponible, se recalcula aqui mismo.

if "Delta Time (s)" in df.columns:
    duracion_real = df["Delta Time (s)"].astype(float)
else:
    duracion_real = (
        df["End Time (s)"].astype(float) - df["Begin Time (s)"].astype(float)
    )

df["Duracion_real_s"] = duracion_real


# =====================================================
# Comparacion contra la ventana fija
# =====================================================

excede_ventana = df["Duracion_real_s"] > DURACION_OBJETIVO
cantidad_excede = int(excede_ventana.sum())
porcentaje_excede = 100 * cantidad_excede / len(df)

print(f"Pulsos que EXCEDEN la ventana fija de {DURACION_OBJETIVO*1000:.1f} ms:")
print(f"  Cantidad   : {cantidad_excede} de {len(df)} ({porcentaje_excede:.2f}%)")
print()

if cantidad_excede > 0:
    exceso_ms = (df.loc[excede_ventana, "Duracion_real_s"] - DURACION_OBJETIVO) * 1000

    print("Magnitud del exceso (cuanto se recorta del pulso), en ms:")
    print(f"  Promedio   : {exceso_ms.mean():.3f} ms")
    print(f"  Mediana    : {exceso_ms.median():.3f} ms")
    print(f"  Maximo     : {exceso_ms.max():.3f} ms")
    print(f"  Percentil 95: {exceso_ms.quantile(0.95):.3f} ms")
    print()


# =====================================================
# Desglose por especie
# =====================================================

print("------------------------------")
print("DESGLOSE POR ESPECIE")
print("------------------------------")

resumen_especie = (
    df.groupby("Specie")
    .apply(
        lambda g: pd.Series({
            "total_pulsos": len(g),
            "exceden_ventana": int((g["Duracion_real_s"] > DURACION_OBJETIVO).sum()),
            "pct_exceden": round(
                100 * (g["Duracion_real_s"] > DURACION_OBJETIVO).sum() / len(g), 2
            ),
            "duracion_promedio_ms": round(g["Duracion_real_s"].mean() * 1000, 3),
            "duracion_max_ms": round(g["Duracion_real_s"].max() * 1000, 3),
        }),
        include_groups=False,
    )
    .sort_values("exceden_ventana", ascending=False)
)

print(resumen_especie.to_string())


# =====================================================
# Distribucion de cuanto excede (para decidir margen de seguridad)
# =====================================================
# Util para decidir, por ejemplo, una ventana fija mas amplia (ej. 20ms
# o 25ms en vez de 15ms) que cubra un mayor porcentaje de pulsos sin
# necesidad de ventana variable.

print()
print("------------------------------")
print("¿QUE VENTANA FIJA CUBRIRIA QUE PORCENTAJE DE PULSOS?")
print("------------------------------")

ventanas_candidatas_ms = [15, 18, 20, 22, 25, 28, 30, 32]

for ventana_ms in ventanas_candidatas_ms:
    ventana_s = ventana_ms / 1000
    cubiertos = int((df["Duracion_real_s"] <= ventana_s).sum())
    pct_cubiertos = 100 * cubiertos / len(df)
    print(
        f"  Ventana {ventana_ms:4d} ms -> cubre {cubiertos:5d}/{len(df)} "
        f"pulsos ({pct_cubiertos:.2f}%)"
    )


# =====================================================
# Guardar detalle de los pulsos problematicos
# =====================================================

RUTA_SALIDA_DETALLE = (
    "/media/adalab/bodega/CamiloBuitrago/"
    "pulsos_exceden_ventana.csv"
)

if cantidad_excede > 0:
    columnas_detalle = [
        c for c in [
            "Begin File", "Specie", "Pulse Type", "Calidad",
            "Begin Time (s)", "End Time (s)", "Duracion_real_s", "Pulse_ID",
        ]
        if c in df.columns
    ]
    df.loc[excede_ventana, columnas_detalle].to_csv(
        RUTA_SALIDA_DETALLE, index=False
    )
    print()
    print("Detalle de pulsos que exceden la ventana guardado en:")
    print(RUTA_SALIDA_DETALLE)

print()
print("==============================")
print("AUDITORIA FINALIZADA")
print("==============================")
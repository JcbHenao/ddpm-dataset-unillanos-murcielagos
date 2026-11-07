"""
FASE 3 - AUDITORIA DEL RESULTADO NORMALIZADO
=====================================================
Fusiona: audit_pulsos.py, audit_ventana_pulsos.py y
audit_columnas_categorias.py, todos apuntando ahora al UNICO archivo
maestro (config.ARCHIVO_MAESTRO) en vez de al CSV viejo
"anotaciones_pulsos_limpias.csv".

Diferencia importante frente a los scripts viejos: el archivo maestro
ahora mezcla Pulsos (decenas de ms) y Secuencias (varios segundos). Si
se calculan las estadisticas de duracion sobre las dos cosas juntas,
la duracion "promedio" de una especie queda inflada por las secuencias
y dice poco sobre el pulso individual. Por eso esta fase SIEMPRE separa
por Tipo_Anotacion antes de calcular duracion/ventana, y solo evalua la
ventana fija de generacion de espectrogramas (fase4) contra las filas
de tipo "Pulso" (las Secuencias no se les genera espectrograma de
ventana fija).
=====================================================
"""

import os
import difflib
import pandas as pd

import config
from espectrogramas_secuencias_comun import generar_ventanas

# Mas estricto que en fase1 (0.6): aqui se compara sobre categorias YA
# normalizadas, que deberian ser pocas y limpias. Con 0.6, nombres de
# especie que comparten prefijo largo a proposito (ej. "MolossusMolossus"
# vs "MolossusRufus" vs "Molossus_1", que SI son especies/variantes
# distintas, no typos entre si) generaban falsos positivos.
UMBRAL_SIMILITUD = 0.85


def similitud(a, b):
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


def agrupar_por_similitud(valores_con_conteo, umbral):
    items = sorted(valores_con_conteo.items(), key=lambda x: -x[1])
    asignado = set()
    grupos = []
    for valor, conteo in items:
        if valor in asignado:
            continue
        grupo = [(valor, conteo)]
        asignado.add(valor)
        for otro_valor, otro_conteo in items:
            if otro_valor in asignado:
                continue
            if similitud(valor, otro_valor) >= umbral:
                grupo.append((otro_valor, otro_conteo))
                asignado.add(otro_valor)
        grupos.append(grupo)
    return grupos


def seccion_duracion_y_frecuencias(df, etiqueta):
    print(f"\n--- DURACION Y FRECUENCIAS: {etiqueta} ({len(df)} filas) ---")
    if df.empty:
        print("  (sin filas)")
        return

    for nombre, valor in {
        "Promedio": df["Delta Time (s)"].mean(),
        "Mediana": df["Delta Time (s)"].median(),
        "Minimo": df["Delta Time (s)"].min(),
        "Maximo": df["Delta Time (s)"].max(),
        "Percentil 95": df["Delta Time (s)"].quantile(0.95),
    }.items():
        print(f"  {nombre:15}: {valor:.6f} s")

    print(f"  Peak Freq promedio: {df['Peak Freq (Hz)'].mean():.1f} Hz")
    print(f"  Low Freq minimo   : {df['Low Freq (Hz)'].min():.1f} Hz")
    print(f"  High Freq maximo  : {df['High Freq (Hz)'].max():.1f} Hz")

    tabla = (
        df.groupby("Specie")["Delta Time (s)"]
        .agg(cantidad="count", promedio="mean", mediana="median", maximo="max")
        .sort_values("cantidad", ascending=False)
    )
    print(f"\n  Por especie:\n{tabla.to_string()}")


def seccion_ventana_fija(df_pulsos):
    print(f"\n--- VENTANA FIJA vs DURACION REAL (solo Tipo_Anotacion=='Pulso') ---")
    print(f"Ventana fija configurada: {config.DURACION_OBJETIVO_VENTANA*1000:.1f} ms")

    excede = df_pulsos["Delta Time (s)"] > config.DURACION_OBJETIVO_VENTANA
    n_excede = int(excede.sum())
    print(f"Pulsos que exceden la ventana: {n_excede}/{len(df_pulsos)} ({100*n_excede/len(df_pulsos):.2f}%)")

    print("\n¿Que ventana fija cubriria que porcentaje de pulsos?")
    for ventana_ms in [15, 18, 20, 22, 25, 28, 30, 32]:
        ventana_s = ventana_ms / 1000
        cubiertos = int((df_pulsos["Delta Time (s)"] <= ventana_s).sum())
        pct = 100 * cubiertos / len(df_pulsos)
        print(f"  {ventana_ms:4d} ms -> cubre {cubiertos:5d}/{len(df_pulsos)} ({pct:.2f}%)")

    if n_excede:
        ruta = os.path.join(config.SALIDA_FASE3, "pulsos_exceden_ventana.csv")
        cols = [c for c in ["Begin File", "Specie", "Pulse Type", "Calidad",
                             "Begin Time (s)", "End Time (s)", "Delta Time (s)", "Anotacion_ID"]
                if c in df_pulsos.columns]
        df_pulsos.loc[excede, cols].to_csv(ruta, index=False)
        print(f"\nDetalle de pulsos que exceden la ventana: {ruta}")


def seccion_tramos_secuencias(df_secuencias):
    print(f"\n--- TRAMOS POR SECUENCIA (Tipo_Anotacion=='Secuencia', chunk={config.DURACION_CHUNK_SECUENCIA:.1f}s) ---")
    print("Cada secuencia se corta en 1 o mas tramos de duracion fija (ver generar_ventanas en")
    print("espectrogramas_secuencias_comun.py): igual a la duracion del chunk -> 1 tramo; entre 1")
    print("chunk y 4.5s -> 3 tramos (inicio/centro/final, se solapan); >4.5s -> consecutivos + 1")
    print("final sobre el sobrante; <1 chunk -> 1 tramo centrado.")

    n_tramos_por_fila = df_secuencias.apply(
        lambda fila: len(generar_ventanas(
            float(fila["Begin Time (s)"]), float(fila["End Time (s)"]), config.DURACION_CHUNK_SECUENCIA
        )),
        axis=1,
    )
    total_tramos = int(n_tramos_por_fila.sum())
    print(f"\nSecuencias: {len(df_secuencias)}  ->  tramos/imagenes totales que generara fase5: {total_tramos}")

    distribucion = n_tramos_por_fila.value_counts().sort_index()
    print("\nDistribucion de tramos por secuencia:")
    for n_tramos, cantidad in distribucion.items():
        print(f"  {n_tramos} tramo(s) -> {cantidad} secuencias")


def seccion_categorias_residuales(df):
    print("\n--- CATEGORIAS RESIDUALES (variantes tipograficas que sobrevivieron al mapeo) ---")
    filas_reporte = []
    for columna in ["Specie", "Calidad", "Pulse Type"]:
        conteo = df[columna].dropna().astype(str).value_counts().to_dict()
        grupos = agrupar_por_similitud(conteo, UMBRAL_SIMILITUD)
        variantes_encontradas = [g for g in grupos if len(g) > 1]
        if not variantes_encontradas:
            continue
        print(f"\n  Columna '{columna}':")
        for grupo in variantes_encontradas:
            canonico, n_canonico = grupo[0]
            print(f"    '{canonico}' ({n_canonico}) <-> variantes: {[v for v, _ in grupo[1:]]}")
            for variante, n in grupo[1:]:
                filas_reporte.append({
                    "columna": columna, "valor_frecuente": canonico,
                    "posible_variante": variante, "conteo_variante": n,
                })
    if filas_reporte:
        ruta = os.path.join(config.SALIDA_FASE3, "categorias_residuales.csv")
        pd.DataFrame(filas_reporte).to_csv(ruta, index=False)
        print(f"\n  Si esto NO esta vacio, el archivo maestro deberia estar 100% "
              f"normalizado -- estas variantes indican que falta una entrada "
              f"en mapeo_*.csv (fase2 no debio haber generado categorias "
              f"residuales, revisar). Detalle: {ruta}")
    else:
        print("  Ninguna. Las categorias del archivo maestro estan limpias.")


def main():
    config.crear_carpetas_salida()

    print("=" * 60)
    print("FASE 3 - AUDITORIA DEL RESULTADO NORMALIZADO")
    print("=" * 60)

    df = pd.read_csv(config.ARCHIVO_MAESTRO)
    for col in ["Delta Time (s)", "Low Freq (Hz)", "High Freq (Hz)", "Peak Freq (Hz)"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"Total filas en archivo maestro: {len(df)}")
    print(df["Tipo_Anotacion"].value_counts().to_string())

    df_pulsos = df[df["Tipo_Anotacion"] == "Pulso"]
    df_secuencias = df[df["Tipo_Anotacion"] == "Secuencia"]

    seccion_duracion_y_frecuencias(df_pulsos, "PULSOS")
    seccion_duracion_y_frecuencias(df_secuencias, "SECUENCIAS")
    seccion_ventana_fija(df_pulsos)
    seccion_tramos_secuencias(df_secuencias)
    seccion_categorias_residuales(df)

    print("\n" + "=" * 60)
    print("FASE 3 FINALIZADA")
    print("=" * 60)


if __name__ == "__main__":
    main()

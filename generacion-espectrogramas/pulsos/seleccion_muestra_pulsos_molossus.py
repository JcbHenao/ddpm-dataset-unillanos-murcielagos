import os
import shutil
import pandas as pd


# =====================================================
# Configuracion
# =====================================================

RUTA_METADATOS = (
    "/media/adalab/bodega/CamiloBuitrago/"
    "Espectrogramas_Pulsos_Anotados_128px/metadatos.csv"
)

RUTA_ESPECTROGRAMAS = (
    "/media/adalab/bodega/CamiloBuitrago/"
    "Espectrogramas_Pulsos_Anotados_128px"
)

RUTA_SALIDA_MUESTRA = (
    "/media/adalab/bodega/CamiloBuitrago/"
    "Muestra_Molossus_Alta_Calidad_128"
)

# Especies a combinar (las 3 variantes de Molossus), en orden alfabetico.
# El orden importa: cuando el total no se divide exacto entre las 3
# (ej. 1000/3 = 333.33), las primeras especies de esta lista son las
# que reciben 1 unidad extra, de forma determinista y reproducible.
ESPECIES_OBJETIVO = sorted([
    "MolossusMolossus",
    "MolossusRufus",
    "Molossus_1",
])

CALIDAD_OBJETIVO = "Alta"

# Tope exacto de imagenes a copiar EN TOTAL, repartido en partes
# iguales entre las especies de ESPECIES_OBJETIVO (no generar de mas).
MAXIMO_IMAGENES_TOTAL = 1000

# Semilla fija para que la seleccion sea reproducible si se vuelve a correr
SEMILLA = 42


os.makedirs(RUTA_SALIDA_MUESTRA, exist_ok=True)


# =====================================================
# Cargar metadatos
# =====================================================

df = pd.read_csv(RUTA_METADATOS)

print("==============================")
print("SELECCION DE MUESTRA BALANCEADA: Molossus, Calidad Alta")
print("==============================")
print("Total filas en metadatos:", len(df))


# =====================================================
# Filtrar por especie + calidad
# =====================================================

mask_especie = df["especie"].isin(ESPECIES_OBJETIVO)
mask_calidad = df["calidad"] == CALIDAD_OBJETIVO

candidatos = df[mask_especie & mask_calidad].copy()

disponibles_por_especie = (
    candidatos["especie"].value_counts().reindex(ESPECIES_OBJETIVO).fillna(0).astype(int)
)

print()
print(f"Pulsos Molossus con Calidad='{CALIDAD_OBJETIVO}', por especie:")
print(disponibles_por_especie.to_string())
print(f"Total disponible (3 especies combinadas): {len(candidatos)}")


# =====================================================
# Calcular cuota balanceada por especie
# =====================================================
# Reparto en partes iguales. Si MAXIMO_IMAGENES_TOTAL no es divisible
# exacto entre el numero de especies, las primeras especies (en orden
# alfabetico) reciben 1 unidad extra para completar el total exacto.
# Ej: 1000 / 3 especies -> 333, 333, 334.

n_especies = len(ESPECIES_OBJETIVO)
cuota_base = MAXIMO_IMAGENES_TOTAL // n_especies
resto = MAXIMO_IMAGENES_TOTAL % n_especies

cuotas = {}
for i, especie in enumerate(ESPECIES_OBJETIVO):
    cuotas[especie] = cuota_base + (1 if i < resto else 0)

print()
print("Cuota balanceada objetivo por especie:")
for especie, cuota in cuotas.items():
    print(f"  {especie:20} -> {cuota}")


# =====================================================
# Verificar si alguna especie no alcanza su cuota
# =====================================================
# Si una especie tiene menos imagenes disponibles que su cuota, no se
# completa con otra especie (eso rompe el balance). En su lugar, se
# reduce el total: la cuota final de TODAS las especies pasa a ser el
# minimo entre "cuota objetivo" y "lo que tenga la especie mas limitada
# entre las que no alcanzan", manteniendo el balance perfecto aunque el
# total final quede por debajo de MAXIMO_IMAGENES_TOTAL.

especies_insuficientes = {
    especie: disponibles_por_especie[especie]
    for especie in ESPECIES_OBJETIVO
    if disponibles_por_especie[especie] < cuotas[especie]
}

if especies_insuficientes:
    cuota_ajustada = min(especies_insuficientes.values())

    print()
    print("AVISO: las siguientes especies no alcanzan su cuota balanceada:")
    for especie, disponible in especies_insuficientes.items():
        print(
            f"  {especie:20} -> disponibles {disponible}, "
            f"cuota objetivo {cuotas[especie]}"
        )
    print(
        f"\nPara mantener el balance perfecto, la cuota de TODAS las "
        f"especies se reduce a {cuota_ajustada} (la del minimo disponible "
        f"entre las especies insuficientes)."
    )
    print(
        f"Total final: {cuota_ajustada * n_especies} en vez de "
        f"{MAXIMO_IMAGENES_TOTAL} (no se completa con otra especie "
        f"para no romper el balance)."
    )

    cuotas = {especie: cuota_ajustada for especie in ESPECIES_OBJETIVO}


# =====================================================
# Seleccionar muestra balanceada (misma cuota exacta por especie)
# =====================================================

partes = []
for especie in ESPECIES_OBJETIVO:
    cuota = cuotas[especie]
    pool_especie = candidatos[candidatos["especie"] == especie]

    seleccion_especie = pool_especie.sample(
        n=cuota, random_state=SEMILLA
    )
    partes.append(seleccion_especie)

muestra = pd.concat(partes, ignore_index=True)

print()
print("Composicion final de la muestra balanceada:")
print(muestra["especie"].value_counts().reindex(ESPECIES_OBJETIVO).to_string())
print(f"Total en la muestra: {len(muestra)}")


# =====================================================
# Copiar archivos
# =====================================================

copiados = 0
no_encontrados = []

for _, fila in muestra.iterrows():
    nombre_archivo = fila["archivo_imagen"]
    especie = fila["especie"]

    ruta_origen = os.path.join(RUTA_ESPECTROGRAMAS, especie, nombre_archivo)

    if not os.path.exists(ruta_origen):
        no_encontrados.append(ruta_origen)
        continue

    ruta_destino = os.path.join(RUTA_SALIDA_MUESTRA, nombre_archivo)
    shutil.copy2(ruta_origen, ruta_destino)
    copiados += 1

print()
print("------------------------------")
print("RESULTADO")
print("------------------------------")
print(f"Imagenes copiadas: {copiados} / {len(muestra)} solicitadas")

if no_encontrados:
    print(f"Imagenes no encontradas en disco: {len(no_encontrados)}")
    ruta_log_faltantes = os.path.join(
        RUTA_SALIDA_MUESTRA, "_faltantes_no_copiadas.csv"
    )
    pd.DataFrame({"ruta_esperada": no_encontrados}).to_csv(
        ruta_log_faltantes, index=False
    )
    print(f"Listado de faltantes guardado en: {ruta_log_faltantes}")

# Guardar tambien el CSV de la muestra seleccionada, para trazabilidad
ruta_csv_muestra = os.path.join(
    RUTA_SALIDA_MUESTRA, "_muestra_seleccionada.csv"
)
muestra.to_csv(ruta_csv_muestra, index=False)
print(f"CSV de la muestra seleccionada: {ruta_csv_muestra}")

print()
print("==============================")
print("SELECCION FINALIZADA")
print("==============================")
import difflib
import csv
from collections import defaultdict

import pandas as pd


# =====================================================
# Configuracion
# =====================================================

RUTA_ANOTACIONES = (
    "/media/adalab/bodega/CamiloBuitrago/"
    "Generacion_Espectrogramas/Limpieza_Anotaciones_Pulsos/"
    "anotaciones_pulsos_limpias.csv"
)

RUTA_SALIDA_AUDITORIA = (
    "/media/adalab/bodega/CamiloBuitrago/"
    "Generacion_Espectrogramas/Limpieza_Anotaciones_Pulsos/"
    "auditoria_columnas_categoricas.csv"
)

# Columnas categoricas a auditar
COLUMNAS_A_AUDITAR = ["Pulse Type", "Calidad", "Specie"]

# Umbral de similitud (0-1) para sugerir que dos valores son la misma
# categoria mal escrita DENTRO de la misma columna. 0.6 es permisivo, 0.8 es estricto.
UMBRAL_SIMILITUD = 0.6

# Umbral de similitud para detectar CONTAMINACION CRUZADA entre columnas
# (ej. un valor de Specie que aparecio por error en Pulse Type).
# Mas bajo que el anterior porque las abreviaturas/contracciones de nombres
# largos (ej. "Molcfruf" vs "MolossusRufus") tienen menor similitud de texto
# que los simples typos dentro de una misma categoria corta.
UMBRAL_SIMILITUD_CONTAMINACION = 0.45

# Bloque contiguo minimo de caracteres compartidos para validar una
# sospecha de contaminacion cruzada. Sin este filtro, la metrica de
# subsecuencia puede dar falsos positivos con coincidencias dispersas
# al azar (ej. "Alto" disperso dentro de "MolossusRufus").
MIN_BLOQUE_CONTIGUO_CONTAMINACION = 3

# Cuantas filas de ejemplo mostrar por cada valor raro
N_EJEMPLOS = 3


# =====================================================
# Utilidades
# =====================================================

def similitud(a, b):
    """Similitud de texto 0-1 usando difflib (sin dependencias externas)."""
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


def mayor_bloque_contiguo(a, b):
    """
    Longitud del mayor bloque de caracteres contiguos compartido entre
    a y b (case-insensitive). Sirve para filtrar falsos positivos de
    es_subsecuencia_aproximada, que por si sola puede dar score alto
    con coincidencias dispersas al azar (ej. 'Alto' disperso dentro de
    'MolossusRufus' sin que las letras esten realmente relacionadas).
    """
    matcher = difflib.SequenceMatcher(None, a.lower(), b.lower())
    bloque = matcher.find_longest_match(0, len(a), 0, len(b))
    return bloque.size


def es_subsecuencia_aproximada(corto, largo, tolerancia_saltos=2):
    """
    Revisa si la cadena 'corto' aparece como subsecuencia (en orden,
    no necesariamente contigua) dentro de 'largo', permitiendo hasta
    'tolerancia_saltos' caracteres de 'corto' que no se encuentren.
    Esto detecta abreviaturas/contracciones como 'Molcfruf' dentro de
    'MolossusRufus' (Mol + c + fruf), que difflib por si solo no detecta
    bien porque no son substrings contiguos.
    Devuelve un score 0-1: proporcion de caracteres de 'corto' que se
    encontraron en orden dentro de 'largo'.
    NOTA: este score por si solo puede dar falsos positivos con
    coincidencias dispersas al azar. Combinar siempre con
    mayor_bloque_contiguo() como filtro adicional.
    """
    corto = corto.lower()
    largo = largo.lower()

    if len(corto) == 0 or len(largo) == 0:
        return 0.0

    pos_largo = 0
    encontrados = 0

    for ch in corto:
        idx = largo.find(ch, pos_largo)
        if idx == -1:
            continue
        encontrados += 1
        pos_largo = idx + 1

    return encontrados / len(corto)


def agrupar_por_similitud(valores_con_conteo, umbral):
    """
    Agrupa valores categoricos por similitud de texto.
    valores_con_conteo: dict {valor: conteo}
    Devuelve: lista de grupos, cada grupo es lista de (valor, conteo),
    ordenados por conteo descendente (el primero del grupo = forma "canonica" sugerida).
    """
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


def detectar_contaminacion_cruzada(df, columnas):
    """
    Para cada columna auditada, revisa si alguno de sus valores unicos
    coincide (similitud alta) con valores unicos de OTRA columna auditada.
    Esto detecta casos como 'Molcfruf' apareciendo en Pulse Type cuando
    en realidad es un valor de Specie.
    """
    valores_por_columna = {
        col: set(df[col].dropna().astype(str).unique())
        for col in columnas
    }

    contaminaciones = []

    for col_origen in columnas:
        for valor in valores_por_columna[col_origen]:
            for col_destino in columnas:
                if col_destino == col_origen:
                    continue
                for valor_destino in valores_por_columna[col_destino]:
                    sim_directa = similitud(valor, valor_destino)

                    # Probar subsecuencia en ambas direcciones (corto en
                    # largo y largo en corto) para no asumir cual es cual.
                    if len(valor) <= len(valor_destino):
                        sim_subsecuencia = es_subsecuencia_aproximada(valor, valor_destino)
                    else:
                        sim_subsecuencia = es_subsecuencia_aproximada(valor_destino, valor)

                    sim_final = max(sim_directa, sim_subsecuencia)
                    bloque = mayor_bloque_contiguo(valor, valor_destino)

                    if (
                        sim_final >= UMBRAL_SIMILITUD_CONTAMINACION
                        and bloque >= MIN_BLOQUE_CONTIGUO_CONTAMINACION
                    ):
                        contaminaciones.append({
                            "valor_sospechoso": valor,
                            "columna_donde_aparece": col_origen,
                            "valor_similar_en_otra_columna": valor_destino,
                            "columna_de_origen_probable": col_destino,
                            "similitud": round(sim_final, 2),
                            "bloque_contiguo": bloque,
                        })

    # Ordenar por similitud descendente y quedarse con el mejor candidato
    # por cada (valor_sospechoso, columna_donde_aparece), para no saturar
    # el reporte con candidatos secundarios menos relevantes.
    contaminaciones.sort(key=lambda c: -c["similitud"])

    mejores_por_valor = {}
    for c in contaminaciones:
        clave = (c["valor_sospechoso"], c["columna_donde_aparece"])
        if clave not in mejores_por_valor:
            mejores_por_valor[clave] = c

    contaminaciones = list(mejores_por_valor.values())

    return contaminaciones


def obtener_ejemplos(df, columna, valor, n=N_EJEMPLOS):
    filas = df[df[columna].astype(str) == str(valor)].head(n)
    columnas_contexto = [
        c for c in ["Begin File", "Specie", "Pulse Type", "Calidad", "Pulse_ID"]
        if c in df.columns
    ]
    return filas[columnas_contexto].to_dict("records")


# =====================================================
# Main
# =====================================================

def main():
    df = pd.read_csv(RUTA_ANOTACIONES)

    print("==============================")
    print("AUDITORIA DE COLUMNAS CATEGORICAS")
    print("==============================")
    print("Total filas:", len(df))
    print()

    filas_reporte = []

    # -------------------------------------------------
    # 1. Agrupacion por similitud, columna por columna
    # -------------------------------------------------
    for columna in COLUMNAS_A_AUDITAR:
        if columna not in df.columns:
            print(f"[AVISO] Columna '{columna}' no existe en el CSV, se omite.")
            continue

        print(f"\n--- Columna: {columna} ---")

        conteo = df[columna].dropna().astype(str).value_counts().to_dict()
        print(f"Valores unicos: {len(conteo)} | Total no nulos: {sum(conteo.values())}")

        grupos = agrupar_por_similitud(conteo, UMBRAL_SIMILITUD)

        print(f"Grupos sugeridos por similitud: {len(grupos)}")
        print()

        for grupo in grupos:
            canonico, conteo_canonico = grupo[0]
            variantes = grupo[1:]

            print(f"  Forma mas frecuente: '{canonico}' ({conteo_canonico} filas)")

            for variante, conteo_variante in variantes:
                ejemplos = obtener_ejemplos(df, columna, variante)
                print(f"    -> posible variante: '{variante}' ({conteo_variante} filas)")

                filas_reporte.append({
                    "columna": columna,
                    "valor_canonico_sugerido": canonico,
                    "valor_variante": variante,
                    "conteo_variante": conteo_variante,
                    "tipo_hallazgo": "variante_tipografica",
                    "ejemplos": str(ejemplos),
                })

            if not variantes:
                filas_reporte.append({
                    "columna": columna,
                    "valor_canonico_sugerido": canonico,
                    "valor_variante": "",
                    "conteo_variante": "",
                    "tipo_hallazgo": "sin_variantes",
                    "ejemplos": "",
                })

    # -------------------------------------------------
    # 2. Deteccion de contaminacion cruzada entre columnas
    # -------------------------------------------------
    print("\n\n--- CONTAMINACION CRUZADA ENTRE COLUMNAS ---")
    print("(valores que parecen pertenecer a OTRA columna)\n")

    columnas_validas = [c for c in COLUMNAS_A_AUDITAR if c in df.columns]
    contaminaciones = detectar_contaminacion_cruzada(df, columnas_validas)

    if not contaminaciones:
        print("No se detecto contaminacion cruzada evidente.")
    else:
        for c in contaminaciones:
            print(
                f"  '{c['valor_sospechoso']}' en columna '{c['columna_donde_aparece']}' "
                f"se parece a '{c['valor_similar_en_otra_columna']}' "
                f"de la columna '{c['columna_de_origen_probable']}' "
                f"(similitud {c['similitud']}, bloque contiguo {c['bloque_contiguo']})"
            )

            ejemplos = obtener_ejemplos(
                df, c["columna_donde_aparece"], c["valor_sospechoso"]
            )

            filas_reporte.append({
                "columna": c["columna_donde_aparece"],
                "valor_canonico_sugerido": c["valor_similar_en_otra_columna"],
                "valor_variante": c["valor_sospechoso"],
                "conteo_variante": "",
                "tipo_hallazgo": (
                    f"posible_contaminacion_cruzada_desde_{c['columna_de_origen_probable']}"
                ),
                "ejemplos": str(ejemplos),
            })

    # -------------------------------------------------
    # 3. Exportar reporte a CSV
    # -------------------------------------------------
    if filas_reporte:
        campos = list(filas_reporte[0].keys())
        with open(RUTA_SALIDA_AUDITORIA, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=campos)
            writer.writeheader()
            writer.writerows(filas_reporte)

        print(f"\n\nReporte de auditoria guardado en:\n{RUTA_SALIDA_AUDITORIA}")

    print("\n==============================")
    print("AUDITORIA FINALIZADA")
    print("==============================")


if __name__ == "__main__":
    main()
"""
FASE 1 - AUDITORIA DE CATEGORIAS CRUDAS (antes de normalizar)
=====================================================
Reemplaza y fusiona: audit_anotacion_especie.py, audit_anotacion_calidad.py,
audit_anotacion_pulsetype.py (los 3 hacian el mismo recorrido de
carpetas por separado, uno por columna).

Diferencia clave con los scripts viejos:
  Los 3 audit_anotacion_*.py originales solo leian RUTA_ANOTACIONES =
  ".../Anotaciones/Pulsos". Nunca miraban "Anotaciones/Secuencias", asi
  que nunca hubieran detectado que la columna "Pulse Type" en los
  archivos de Secuencias tiene valores propios de ese contexto
  ("Secuencia", "Secuenci") que no son typos de Bajo/Medio/Alto sino
  una categoria distinta (la columna casi no aporta info a nivel de
  secuencia). Esta fase audita Pulsos + Secuencias juntos.

Que hace:
  1. Lee la fuente canonica completa (Pulsos + Secuencias, todos los
     anotadores).
  2. Para Specie y Calidad: compara los valores crudos contra los
     diccionarios de mapeo actuales y separa:
       - cubiertos_por_mapeo   : ya tienen una entrada en el diccionario
       - pendientes_revision   : el anotador marco explicitamente
                                  "Revisar..." (no es un typo, es una
                                  bandera intencional)
       - no_cubiertos          : no estan en el diccionario y no son
                                  "Revisar..." -> necesitan que alguien
                                  decida si es un typo nuevo o una
                                  categoria nueva legitima
     Para Pulse Type se hace lo mismo PERO solo sobre filas de tipo
     "Pulso": fase2 nunca usa mapeo_pulse_type.csv para filas de tipo
     "Secuencia" (las fuerza directo a "No_aplica_secuencia"), asi que
     pedir que se agregue "Secuencia"/"Secuenci" a ese diccionario no
     tiene sentido y solo generaba ruido. Los valores crudos de Pulse
     Type en filas de Secuencia se reportan aparte, en un bloque
     informativo: se espera que casi todos digan "Secuencia"/"Secuenci"
     (redundante con Tipo_Anotacion, no requiere accion); lo que SI se
     destaca ahi son los que no dicen eso (ej. "Regular", "Slep
     quizas"), porque son datos mal capturados en el .txt original,
     aunque fase2 los descarte igual al forzar "No_aplica_secuencia".
  3. Sobre los "no_cubiertos", corre agrupacion por similitud de texto
     (heredado de audit_columnas_categorias.py) para sugerir a que
     valor existente podrian corresponder.
  4. Detecta contaminacion cruzada entre columnas (ej. un valor de
     Specie que aparecio por error en Pulse Type). Un valor solo se
     considera "sospechoso" si NO es un valor valido dentro de su
     propia columna -- si no, "Bajo"/"Medio"/"Alto" (validos en Pulse
     Type) y "Baja"/"Media"/"Alta" (validos en Calidad) se marcaban
     como contaminacion mutua por el simple hecho de sonar parecido,
     cuando en realidad las dos columnas comparten ese vocabulario por
     diseno y no es un error de captura. Con este filtro, lo que queda
     visible es lo que si importa: nombres de especie colados en Pulse
     Type (Molcfruf) y valores de Calidad colados en Pulse Type
     (Regular, que no existe como categoria valida de Pulse Type).

Salida: Fase1_Auditoria_Categorias_Crudas/valores_no_cubiertos.csv,
        pendientes_revision.csv, contaminacion_cruzada.csv y
        pulse_type_en_secuencias.csv (informativo).
Los tres primeros son la lista de trabajo para actualizar los
mapeo_*.csv antes de correr fase2.
=====================================================
"""

import difflib
import pandas as pd

import config
from lector_anotaciones import leer_carpeta_anotaciones, validar_fuente_canonica

COLUMNAS_SIMPLES = ["Specie", "Calidad"]  # se auditan igual para toda fila

UMBRAL_SIMILITUD_SUGERENCIA = 0.6
UMBRAL_SIMILITUD_CONTAMINACION = 0.45
MIN_BLOQUE_CONTIGUO_CONTAMINACION = 3

# Valores de Pulse Type en filas de Secuencia que son el caso esperado
# (la columna solo confirma que es una secuencia) y no ameritan revision.
PULSE_TYPE_SECUENCIA_ESPERADO = {"secuencia", "secuenci"}


def cargar_mapeo(ruta, columna_original, columna_final):
    df = pd.read_csv(ruta, dtype=str)
    df = df.dropna(subset=[columna_original])
    return dict(zip(df[columna_original].str.strip(), df[columna_final].str.strip()))


def es_bandera_revision(valor):
    return str(valor).strip().lower().startswith("revisar")


def similitud(a, b):
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


def mayor_bloque_contiguo(a, b):
    matcher = difflib.SequenceMatcher(None, a.lower(), b.lower())
    bloque = matcher.find_longest_match(0, len(a), 0, len(b))
    return bloque.size


def sugerir_similares(valor, candidatos, umbral=UMBRAL_SIMILITUD_SUGERENCIA):
    sugerencias = [(c, similitud(valor, c)) for c in candidatos]
    sugerencias = [s for s in sugerencias if s[1] >= umbral]
    sugerencias.sort(key=lambda x: -x[1])
    return sugerencias[:3]


def auditar_columna(df, columna, mapeo, filas_no_cubiertas, filas_pendientes_revision):
    """
    Audita una columna completa contra su mapeo y devuelve el conteo de
    valores unicos evaluados (para el resumen impreso). Modifica en
    sitio las listas filas_no_cubiertas / filas_pendientes_revision.
    """
    conteo = df[columna].dropna().astype(str).str.strip()
    conteo = conteo[conteo != ""].value_counts()

    cubiertos = 0
    for valor, n in conteo.items():
        if valor in mapeo:
            cubiertos += 1
            continue
        if es_bandera_revision(valor):
            print(f"  [PENDIENTE REVISION] '{valor}' ({n} filas) -- bandera del anotador, no es typo")
            filas_pendientes_revision.append({"columna": columna, "valor_crudo": valor, "cantidad": n})
            continue

        sugerencias = sugerir_similares(valor, list(mapeo.keys()) + list(mapeo.values()))
        sug_txt = "; ".join(f"{s}({sim:.2f})" for s, sim in sugerencias) or "(sin sugerencia)"
        print(f"  [NO CUBIERTO] '{valor}' ({n} filas) -- posible: {sug_txt}")
        filas_no_cubiertas.append({
            "columna": columna, "valor_crudo": valor, "cantidad": n,
            "sugerencias_por_similitud": sug_txt,
        })

    return len(conteo), cubiertos


def detectar_contaminacion_cruzada(df, columnas, mapeos):
    """
    Compara valores entre columnas para detectar posible contaminacion
    (un dato que quedo en la columna equivocada). Un valor solo entra
    como "sospechoso" si NO esta cubierto por el mapeo de SU PROPIA
    columna -- si ya es un valor valido ahi, la coincidencia con otra
    columna es vocabulario compartido por diseno, no un error de
    captura.
    """
    valores_por_columna = {
        col: set(df[col].dropna().astype(str).str.strip().unique())
        for col in columnas
    }

    hallazgos = []
    for col_origen in columnas:
        mapeo_origen = mapeos.get(col_origen, {})
        for valor in valores_por_columna[col_origen]:
            if not valor or valor in mapeo_origen or es_bandera_revision(valor):
                continue  # valido en su propia columna (o pendiente de revision aparte): no es sospechoso

            for col_destino in columnas:
                if col_destino == col_origen:
                    continue
                for valor_destino in valores_por_columna[col_destino]:
                    if not valor_destino:
                        continue
                    sim = similitud(valor, valor_destino)
                    bloque = mayor_bloque_contiguo(valor, valor_destino)
                    if sim >= UMBRAL_SIMILITUD_CONTAMINACION and bloque >= MIN_BLOQUE_CONTIGUO_CONTAMINACION:
                        hallazgos.append({
                            "valor_sospechoso": valor,
                            "columna_donde_aparece": col_origen,
                            "valor_similar_en": valor_destino,
                            "columna_probable_origen": col_destino,
                            "similitud": round(sim, 2),
                        })

    hallazgos.sort(key=lambda h: -h["similitud"])
    vistos = set()
    unicos = []
    for h in hallazgos:
        clave = (h["valor_sospechoso"], h["columna_donde_aparece"])
        if clave not in vistos:
            vistos.add(clave)
            unicos.append(h)
    return unicos


def main():
    print("=" * 60)
    print("FASE 1 - AUDITORIA DE CATEGORIAS CRUDAS")
    print("=" * 60)

    df_pulsos, _ = leer_carpeta_anotaciones(config.RUTA_PULSOS, "Pulso")
    df_secuencias, _ = leer_carpeta_anotaciones(config.RUTA_SECUENCIAS, "Secuencia")
    if not validar_fuente_canonica(df_pulsos, df_secuencias, config):
        return
    config.crear_carpetas_salida()
    df = pd.concat([df_pulsos, df_secuencias], ignore_index=True)
    df = df.rename(columns={"Grupo_Fuente": "Tipo_Anotacion"})

    print(f"Filas auditadas (Pulsos + Secuencias, todos los anotadores): {len(df)}")

    mapeos = {
        "Specie": cargar_mapeo(config.RUTA_MAPEO_ESPECIES, "Especie_original", "Especie"),
        "Calidad": cargar_mapeo(config.RUTA_MAPEO_CALIDAD, "Calidad_original", "Calidad"),
        "Pulse Type": cargar_mapeo(config.RUTA_MAPEO_PULSE_TYPE, "Pulse_Type_original", "Pulse_Type"),
    }

    filas_no_cubiertas = []
    filas_pendientes_revision = []

    for columna in COLUMNAS_SIMPLES:
        if columna not in df.columns:
            print(f"\n[AVISO] columna '{columna}' no existe en los datos crudos, se omite.")
            continue
        print(f"\n--- {columna} ---")
        n_unicos, cubiertos = auditar_columna(df, columna, mapeos[columna], filas_no_cubiertas, filas_pendientes_revision)
        print(f"  Valores unicos: {n_unicos} | cubiertos por mapeo: {cubiertos}")

    # --- Pulse Type: solo se exige cobertura de mapeo en filas de Pulso ---
    filas_secuencia_pulse_type_raras = []
    if "Pulse Type" in df.columns:
        print("\n--- Pulse Type (solo filas Tipo_Anotacion == 'Pulso') ---")
        df_pt_pulso = df[df["Tipo_Anotacion"] == "Pulso"]
        n_unicos, cubiertos = auditar_columna(
            df_pt_pulso, "Pulse Type", mapeos["Pulse Type"],
            filas_no_cubiertas, filas_pendientes_revision,
        )
        print(f"  Valores unicos: {n_unicos} | cubiertos por mapeo: {cubiertos}")

        print("\n--- Pulse Type en filas Tipo_Anotacion == 'Secuencia' (informativo) ---")
        print("  fase2 fuerza esta columna a 'No_aplica_secuencia' para todas estas filas,")
        print("  sin usar mapeo_pulse_type.csv. Se listan aqui solo para detectar filas mal")
        print("  capturadas en el .txt original (valor distinto de 'Secuencia'/'Secuenci').")
        df_pt_sec = df[df["Tipo_Anotacion"] == "Secuencia"]
        conteo_sec = df_pt_sec["Pulse Type"].dropna().astype(str).str.strip()
        conteo_sec = conteo_sec[conteo_sec != ""].value_counts()
        for valor, n in conteo_sec.items():
            if valor.lower() in PULSE_TYPE_SECUENCIA_ESPERADO:
                print(f"  [OK, esperado] '{valor}' ({n} filas)")
            else:
                print(f"  [DATO RARO EN FILA DE SECUENCIA] '{valor}' ({n} filas) -- revisar el .txt original si importa, "
                      f"fase2 lo descarta igual")
                filas_secuencia_pulse_type_raras.append({"valor_crudo": valor, "cantidad": n})

    print("\n--- CONTAMINACION CRUZADA ENTRE COLUMNAS (datos crudos) ---")
    columnas_validas = [c for c in COLUMNAS_SIMPLES + ["Pulse Type"] if c in df.columns]
    contaminaciones = detectar_contaminacion_cruzada(df, columnas_validas, mapeos)
    if not contaminaciones:
        print("  No se detecto contaminacion cruzada evidente.")
    else:
        for c in contaminaciones:
            print(
                f"  '{c['valor_sospechoso']}' en '{c['columna_donde_aparece']}' "
                f"se parece a '{c['valor_similar_en']}' de '{c['columna_probable_origen']}' "
                f"(similitud {c['similitud']})"
            )

    ruta_no_cubiertos = f"{config.SALIDA_FASE1}/valores_no_cubiertos.csv"
    ruta_pendientes = f"{config.SALIDA_FASE1}/pendientes_revision.csv"
    ruta_contaminacion = f"{config.SALIDA_FASE1}/contaminacion_cruzada.csv"
    ruta_pulse_type_secuencias = f"{config.SALIDA_FASE1}/pulse_type_en_secuencias.csv"

    pd.DataFrame(filas_no_cubiertas).to_csv(ruta_no_cubiertos, index=False)
    pd.DataFrame(filas_pendientes_revision).to_csv(ruta_pendientes, index=False)
    pd.DataFrame(contaminaciones).to_csv(ruta_contaminacion, index=False)
    pd.DataFrame(filas_secuencia_pulse_type_raras).to_csv(ruta_pulse_type_secuencias, index=False)

    print("\n" + "=" * 60)
    print("FASE 1 FINALIZADA")
    print("=" * 60)
    print(f"Valores no cubiertos por los mapeos (accion: decidir y agregar a mapeo_*.csv): {ruta_no_cubiertos}")
    print(f"Valores pendientes de revision del anotador (no son typos, no tocar mapeos): {ruta_pendientes}")
    print(f"Sospechas de contaminacion cruzada entre columnas: {ruta_contaminacion}")
    print(f"Pulse Type raro dentro de filas de Secuencia (informativo, sin accion en mapeos): {ruta_pulse_type_secuencias}")
    print(
        "\nSi 'valores_no_cubiertos.csv' quedo vacio, los mapeos actuales "
        "cubren el 100% de los datos (Specie, Calidad, y Pulse Type de "
        "filas Pulso) y fase2 puede correr sin descartar nada por "
        "categoria desconocida."
    )


if __name__ == "__main__":
    main()

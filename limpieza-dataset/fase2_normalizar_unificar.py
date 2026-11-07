"""
FASE 2 - NORMALIZAR, UNIFICAR Y DEDUPLICAR EN UN SOLO ARCHIVO
=====================================================
Reemplaza: normalizar_anotaciones.py y limpiar_anotaciones_pulsos.py
(este ultimo era una version vieja/incompleta del primero -- mismas
rutas, sin la fusion de Calidad ni el mapeo de Pulse Type ni la
correccion de Molcfruf. Queda retirado, no se vuelve a usar).

Diferencias frente a normalizar_anotaciones.py:

  1. FUENTE: lee Anotaciones/Pulsos Y Anotaciones/Secuencias (el
     script viejo solo leia Pulsos). Total_Anotaciones y
     anotaciones_secuencias_por_audio NO se leen aqui porque fase0
     confirmo que son re-exportaciones redundantes de estos mismos
     datos (ver fase0_inventario_y_redundancia.py). Si fase0 hubiera
     encontrado filas sin gemela, ese es el momento de revisarlas e
     incorporarlas a Pulsos/Secuencias ANTES de correr esta fase.

  2. TRAZABILIDAD: cada fila queda etiquetada con:
       - Tipo_Anotacion : "Pulso" o "Secuencia" (de la carpeta de origen)
       - Anotador       : "Jancel" o "Maira" (de la carpeta de origen)
     validado cruzando contra la columna "Signal Type" del propio txt
     (que deberia decir "pulso"/"secuencia"); si no coincide, se marca
     para revision en vez de asumir cual de las dos fuentes tiene razon.

  3. PULSE TYPE ya no se mapea igual para todo: en las filas de tipo
     "Secuencia", esta columna casi siempre solo dice "Secuencia"
     (redundante con Tipo_Anotacion) y no una categoria Bajo/Medio/
     Alto/corto/largo real. Se guarda el valor crudo en
     Pulse_Type_crudo y se deja Pulse Type = "No_aplica_secuencia"
     para esas filas, en vez de forzarlas por el diccionario de Pulsos.

  4. ESPECIE "Revisar...": el anotador dejo esa marca a proposito para
     decir "todavia no se que especie es". No es un typo, asi que no
     se descarta junto con los valores de especie realmente
     desconocidos: va a un archivo aparte (pendientes_revision.csv)
     para que se resuelva por separado.

  5. ID UNICO: se prefija con el tipo de anotacion (ej.
     "MolossusMolossus_Pulso_000001" vs
     "MolossusMolossus_Secuencia_000001") para que un pulso y la
     secuencia que lo contiene nunca compartan ID aunque sean de la
     misma especie.

  6. SALIDA: un solo archivo (anotaciones_unificadas.csv) con TODAS las
     filas (pulsos + secuencias) y una columna Tipo_Anotacion para
     filtrar. "reunir todo en un solo archivo" sin perder la
     distincion pulso/secuencia, que hace falta para el siguiente paso
     del proyecto (modelo de secuencias).
=====================================================
"""

import os
import pandas as pd
from collections import Counter

import config
from lector_anotaciones import leer_carpeta_anotaciones, validar_fuente_canonica


def cargar_mapeo(ruta, columna_original, columna_final):
    df = pd.read_csv(ruta, dtype=str)
    df = df.dropna(subset=[columna_original])
    return dict(zip(df[columna_original].str.strip(), df[columna_final].str.strip()))


def extraer_anotador(ruta_txt):
    # .../Anotaciones/Pulsos/<Anotador>/<Especie o Sitio>/archivo.txt
    partes = ruta_txt.replace("\\", "/").split("/")
    for etiqueta in ("Pulsos", "Secuencias"):
        if etiqueta in partes:
            idx = partes.index(etiqueta)
            if idx + 1 < len(partes):
                return partes[idx + 1]
    return "Desconocido"


def main():
    print("=" * 60)
    print("FASE 2 - NORMALIZAR, UNIFICAR Y DEDUPLICAR")
    print("=" * 60)

    # -------------------------------------------------
    # 1. Leer fuente canonica (Pulsos + Secuencias)
    # -------------------------------------------------
    df_pulsos, err_p = leer_carpeta_anotaciones(config.RUTA_PULSOS)
    df_secuencias, err_s = leer_carpeta_anotaciones(config.RUTA_SECUENCIAS)

    print(f"Pulsos.txt leidos     : {df_pulsos['Ruta_TXT'].nunique() if not df_pulsos.empty else 0} archivos, {len(df_pulsos)} filas")
    print(f"Secuencias.txt leidos : {df_secuencias['Ruta_TXT'].nunique() if not df_secuencias.empty else 0} archivos, {len(df_secuencias)} filas")

    if not validar_fuente_canonica(df_pulsos, df_secuencias, config):
        return
    config.crear_carpetas_salida()

    df_pulsos["Tipo_Anotacion"] = "Pulso"
    df_secuencias["Tipo_Anotacion"] = "Secuencia"

    errores = err_p + err_s
    if errores:
        print(f"[AVISO] {len(errores)} archivos .txt no se pudieron leer (ver errores_txt.csv)")

    df = pd.concat([df_pulsos, df_secuencias], ignore_index=True)
    print(f"Registros originales (Pulsos + Secuencias): {len(df)}")

    df["Anotador"] = df["Ruta_TXT"].apply(extraer_anotador)

    # -------------------------------------------------
    # 2. Cruce Tipo_Anotacion vs columna "Signal Type" del propio txt
    # -------------------------------------------------
    if "Signal Type" in df.columns:
        tipo_esperado = df["Signal Type"].astype(str).str.strip().str.lower().map(
            {"pulso": "Pulso", "secuencia": "Secuencia"}
        )
        inconsistentes = df["Signal Type"].notna() & (tipo_esperado != df["Tipo_Anotacion"])
        n_inconsistentes = int(inconsistentes.sum())
        if n_inconsistentes:
            print(f"[AVISO] {n_inconsistentes} filas donde la carpeta (Pulsos/Secuencias) "
                  f"no coincide con la columna 'Signal Type' del txt -- revisar")
            df.loc[inconsistentes].to_csv(
                os.path.join(config.SALIDA_FASE2, "inconsistencia_tipo_anotacion.csv"), index=False
            )

    # -------------------------------------------------
    # 3. Cargar mapeos
    # -------------------------------------------------
    mapeo_especies = cargar_mapeo(config.RUTA_MAPEO_ESPECIES, "Especie_original", "Especie")
    mapeo_calidad = cargar_mapeo(config.RUTA_MAPEO_CALIDAD, "Calidad_original", "Calidad")
    mapeo_pulse_type = cargar_mapeo(config.RUTA_MAPEO_PULSE_TYPE, "Pulse_Type_original", "Pulse_Type")

    # -------------------------------------------------
    # 4. Correccion conocida: 'Molcfruf' filtrado por error a Pulse Type
    # -------------------------------------------------
    mask_molcfruf = df["Pulse Type"].astype(str).str.strip() == "Molcfruf"
    filas_molcfruf_corregidas = int(mask_molcfruf.sum())
    # se sobreescribe con el valor CRUDO ("molossus rufus"), para que
    # fluya por el mismo mapeo_especies.csv que todo lo demas, en vez
    # de saltarselo escribiendo directo el valor final normalizado
    df.loc[mask_molcfruf, "Specie"] = "molossus rufus"
    df.loc[mask_molcfruf, "Pulse Type"] = pd.NA

    # -------------------------------------------------
    # 5. Especie: separar "Revisar..." (pendiente) de lo demas
    # -------------------------------------------------
    df["Specie_original"] = df["Specie"].astype(str).str.strip()

    mask_pendiente_especie = df["Specie_original"].str.lower().str.startswith("revisar")
    pendientes_especie = df[mask_pendiente_especie].copy()

    df["Specie"] = df["Specie_original"].map(mapeo_especies)
    mask_especie_ok = df["Specie"].notna() | mask_pendiente_especie
    descartados_especie = df[~mask_especie_ok].copy()

    df = df[~mask_pendiente_especie & df["Specie"].notna()].copy()

    # -------------------------------------------------
    # 6. Calidad: typos -> 5 categorias -> fusion a 3 categorias finales
    # -------------------------------------------------
    df["Calidad_original"] = df["Calidad"].astype(str).str.strip()
    df["Calidad"] = df["Calidad_original"].map(mapeo_calidad).fillna(df["Calidad_original"])

    df["Calidad_5_categorias"] = df["Calidad"]
    df["Calidad"] = df["Calidad_5_categorias"].map(config.FUSION_CALIDAD_FINAL)
    mask_calidad_no_fusionada = df["Calidad"].isna() & df["Calidad_5_categorias"].notna()
    df.loc[mask_calidad_no_fusionada, "Calidad"] = df.loc[mask_calidad_no_fusionada, "Calidad_5_categorias"]
    calidad_no_fusionada = df[mask_calidad_no_fusionada].copy()

    # -------------------------------------------------
    # 7. Pulse Type: solo aplica de verdad a filas de tipo "Pulso".
    #    En filas de tipo "Secuencia", se preserva el valor crudo pero
    #    NO se fuerza por el diccionario de Bajo/Medio/Alto.
    # -------------------------------------------------
    df["Pulse_Type_original"] = df["Pulse Type"].astype(str).str.strip()
    df.loc[df["Pulse Type"].isna(), "Pulse_Type_original"] = pd.NA

    es_pulso = df["Tipo_Anotacion"] == "Pulso"

    df["Pulse Type"] = pd.NA
    df.loc[es_pulso, "Pulse Type"] = df.loc[es_pulso, "Pulse_Type_original"].map(mapeo_pulse_type)
    df.loc[~es_pulso, "Pulse Type"] = "No_aplica_secuencia"

    mask_pulse_type_no_mapeado = es_pulso & df["Pulse Type"].isna() & df["Pulse_Type_original"].notna()
    df.loc[mask_pulse_type_no_mapeado, "Pulse Type"] = df.loc[mask_pulse_type_no_mapeado, "Pulse_Type_original"]
    pulse_type_no_mapeado = df[mask_pulse_type_no_mapeado].copy()

    # -------------------------------------------------
    # 7b. Duracion invalida (ej. Begin Time == End Time -> Delta Time == 0).
    #     Raro pero ocurre (ej. una secuencia de 0 s en los datos actuales);
    #     no se puede generar espectrograma de un evento sin duracion.
    # -------------------------------------------------
    df["Delta Time (s)"] = pd.to_numeric(df["Delta Time (s)"], errors="coerce")
    mask_duracion_invalida = df["Delta Time (s)"].isna() | (df["Delta Time (s)"] <= 0)
    duracion_invalida = df[mask_duracion_invalida].copy()
    df = df[~mask_duracion_invalida].copy()

    # -------------------------------------------------
    # 8. Eliminacion de duplicados
    # -------------------------------------------------
    columnas_id = ["Begin File", "Begin Time (s)", "End Time (s)", "Low Freq (Hz)", "High Freq (Hz)"]
    duplicados = df[df.duplicated(subset=columnas_id, keep=False)].copy()
    df = df.drop_duplicates(subset=columnas_id, keep="first")

    # -------------------------------------------------
    # 9. ID unico, separado por tipo de anotacion
    # -------------------------------------------------
    contador = Counter()
    ids = []
    for especie, tipo in zip(df["Specie"], df["Tipo_Anotacion"]):
        clave = (especie, tipo)
        contador[clave] += 1
        ids.append(f"{especie}_{tipo}_{contador[clave]:06d}")
    df["Anotacion_ID"] = ids

    # -------------------------------------------------
    # 10. Guardar archivo maestro unico + por especie
    # -------------------------------------------------
    df.to_csv(config.ARCHIVO_MAESTRO, index=False, encoding="utf-8")

    for especie, grupo in df.groupby("Specie"):
        grupo.to_csv(
            os.path.join(config.SALIDA_FASE2_POR_ESPECIE, f"{especie}.csv"),
            index=False, encoding="utf-8",
        )

    # -------------------------------------------------
    # 11. Auditorias de la corrida
    # -------------------------------------------------
    duplicados.to_csv(os.path.join(config.SALIDA_FASE2, "duplicados.csv"), index=False)
    descartados_especie.to_csv(os.path.join(config.SALIDA_FASE2, "descartados_especie.csv"), index=False)
    pendientes_especie.to_csv(os.path.join(config.SALIDA_FASE2, "pendientes_revision_especie.csv"), index=False)

    if len(calidad_no_fusionada):
        calidad_no_fusionada.to_csv(os.path.join(config.SALIDA_FASE2, "calidad_no_fusionada.csv"), index=False)
    if len(pulse_type_no_mapeado):
        pulse_type_no_mapeado.to_csv(os.path.join(config.SALIDA_FASE2, "pulse_type_no_mapeado.csv"), index=False)
    if len(duracion_invalida):
        duracion_invalida.to_csv(os.path.join(config.SALIDA_FASE2, "duracion_invalida.csv"), index=False)
    if errores:
        pd.DataFrame(errores).to_csv(os.path.join(config.SALIDA_FASE2, "errores_txt.csv"), index=False)

    resumen_especie = df["Specie"].value_counts().reset_index()
    resumen_especie.columns = ["Especie", "Cantidad"]
    resumen_especie.to_csv(os.path.join(config.SALIDA_FASE2, "resumen_especies.csv"), index=False)

    resumen_tipo = df.groupby(["Specie", "Tipo_Anotacion"]).size().reset_index(name="Cantidad")
    resumen_tipo.to_csv(os.path.join(config.SALIDA_FASE2, "resumen_especie_x_tipo.csv"), index=False)

    # -------------------------------------------------
    # 12. Reporte
    # -------------------------------------------------
    print("\n" + "=" * 60)
    print("FASE 2 FINALIZADA")
    print("=" * 60)
    print(f"Registros finales (unificados, deduplicados): {len(df)}")
    print(f"  Pulsos    : {(df['Tipo_Anotacion']=='Pulso').sum()}")
    print(f"  Secuencias: {(df['Tipo_Anotacion']=='Secuencia').sum()}")
    print(f"Duplicados eliminados: {len(duplicados)}")
    print(f"Descartados por duracion invalida (Delta Time <= 0): {len(duracion_invalida)}")
    print(f"Filas Molcfruf reubicadas a Specie=MolossusRufus: {filas_molcfruf_corregidas}")
    print(f"Descartados por especie no reconocida (typo nuevo, revisar mapeo): {len(descartados_especie)}")
    print(f"Pendientes de revision (especie marcada 'Revisar...' por el anotador): {len(pendientes_especie)}")
    print(f"Calidad no fusionada (revisar mapeo_calidad.csv): {len(calidad_no_fusionada)}")
    print(f"Pulse Type no mapeado en filas de tipo Pulso (revisar mapeo_pulse_type.csv): {len(pulse_type_no_mapeado)}")
    print("\nDistribucion especie x tipo de anotacion:")
    print(resumen_tipo.to_string(index=False))
    print(f"\nArchivo unico final: {config.ARCHIVO_MAESTRO}")


if __name__ == "__main__":
    main()

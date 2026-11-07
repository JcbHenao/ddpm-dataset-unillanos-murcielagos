"""
FASE 5 - GENERACION DE ESPECTROGRAMAS DE SECUENCIAS (anotadas a mano)
=====================================================
Paralela a fase4 (pulsos) y a fase6 (secuencias detectadas por el
modelo FasterRCNN, misma logica de ventana y renderizado -- ver
espectrogramas_secuencias_comun.py -- para que se puedan comparar
"real" vs "modelo" en igualdad de condiciones).

Consume el archivo maestro de fase2/fase3, filtrando
Tipo_Anotacion=="Secuencia". Cada secuencia se corta en una o mas
ventanas de config.DURACION_CHUNK_SECUENCIA (1.5 s por defecto):
igual, entre 1 y 4.5 s, mayor a 4.5 s, o menor a 1 ventana (ver
generar_ventanas en el modulo comun). Nunca se rellena con silencio
sintetico: una ventana que se sale del archivo se recorre para usar
audio real (ver extraer_ventana en el modulo comun). Si el archivo
completo es mas corto que la ventana, no hay audio real con que
completarla -- esa ventana se omite y queda registrada (ver
ventanas_omitidas_audio_corto.csv y secuencias_sin_ninguna_imagen.csv
mas abajo), en vez de solo imprimirse por consola.

Salida: config.SALIDA_ESPECTROGRAMAS_SECUENCIAS, una carpeta por
especie + metadatos.csv + audios_con_nyquist_menor_a_esperado.csv (los
.wav cuyo Nyquist real obligo a recortar HIGH_FREQ) +
ventanas_omitidas_audio_corto.csv (ventanas perdidas por archivo
demasiado corto, con la duracion real disponible) +
secuencias_sin_ninguna_imagen.csv (secuencias que quedaron sin NINGUNA
imagen porque todas sus ventanas cayeron en audios demasiado cortos).
=====================================================
"""

import os
import gc
import csv
import multiprocessing as mp
from collections import defaultdict

import pandas as pd
import librosa

import config
from espectrogramas_secuencias_comun import (
    LOW_FREQ, HIGH_FREQ,
    limpiar_nombre, limpiar_o_marcar_na, indexar_audios,
    generar_ventanas, extraer_ventana, calcular_hop_length,
    render_espectrograma_uniforme,
)

N_PROCESOS = os.cpu_count()


def procesar_audio(args):
    nombre_audio, audio_path, filas = args
    resultados_meta = []
    omitidas_meta = []
    contador_local = defaultdict(int)

    try:
        audio, sr = librosa.load(audio_path, sr=None, mono=True)
    except Exception as e:
        print("Error cargando:", nombre_audio, e)
        return resultados_meta, omitidas_meta

    duracion_audio_disponible = len(audio) / sr
    duracion_ventana = config.DURACION_CHUNK_SECUENCIA
    hop_length, _ = calcular_hop_length(duracion_ventana, sr)

    for fila in filas:
        try:
            inicio_seq = float(fila["Begin Time (s)"])
            fin_seq = float(fila["End Time (s)"])

            ventanas = generar_ventanas(inicio_seq, fin_seq, duracion_ventana)
            n_ventanas = len(ventanas)

            especie = limpiar_nombre(fila["Specie"])
            calidad = limpiar_o_marcar_na(fila.get("Calidad"))
            anotacion_id = fila.get("Anotacion_ID", "")
            audio_base = limpiar_nombre(os.path.splitext(nombre_audio)[0])

            for idx_ventana, (inicio_ventana, _fin_pedida) in enumerate(ventanas):
                muestra, t_inicio, t_fin, ajustada, insuficiente = extraer_ventana(
                    audio, sr, inicio_ventana, duracion_ventana
                )
                if insuficiente:
                    omitidas_meta.append({
                        "audio_origen": nombre_audio,
                        "anotacion_id": anotacion_id,
                        "especie": especie,
                        "ventana_indice": idx_ventana + 1,
                        "ventana_total": n_ventanas,
                        "duracion_ventana_requerida_s": duracion_ventana,
                        "duracion_audio_disponible_s": round(duracion_audio_disponible, 4),
                    })
                    continue

                incluye_contexto = t_inicio < inicio_seq or t_fin > fin_seq

                contador_local[especie] += 1
                numero = contador_local[especie]
                nombre_salida = (
                    f"{audio_base}__t{t_inicio:.3f}-{t_fin:.3f}s"
                    f"__ventana{idx_ventana+1}de{n_ventanas}__{calidad}__s{numero:05d}.png"
                )
                carpeta = os.path.join(config.SALIDA_ESPECTROGRAMAS_SECUENCIAS, especie)
                os.makedirs(carpeta, exist_ok=True)
                ruta = os.path.join(carpeta, nombre_salida)

                high_freq_real = render_espectrograma_uniforme(muestra, sr, hop_length, ruta)

                resultados_meta.append({
                    "archivo_imagen": nombre_salida, "especie": especie,
                    "audio_origen": nombre_audio, "anotacion_id": anotacion_id,
                    "sample_rate_hz": sr,
                    "inicio_secuencia_s": f"{inicio_seq:.6f}", "fin_secuencia_s": f"{fin_seq:.6f}",
                    "ventana_indice": idx_ventana + 1, "ventana_total": n_ventanas,
                    "inicio_ventana_s": f"{t_inicio:.6f}", "fin_ventana_s": f"{t_fin:.6f}",
                    "incluye_audio_fuera_de_la_secuencia": incluye_contexto,
                    "ventana_ajustada_borde_archivo": ajustada,
                    "frecuencia_recortada_por_nyquist": high_freq_real < HIGH_FREQ,
                    "high_freq_usado_hz": high_freq_real,
                    "calidad": calidad,
                })
        except Exception as e:
            print("Error procesando secuencia de", nombre_audio, ":", e)
            continue

    del audio
    gc.collect()
    return resultados_meta, omitidas_meta


def main():
    os.makedirs(config.SALIDA_ESPECTROGRAMAS_SECUENCIAS, exist_ok=True)

    print("=" * 60)
    print("FASE 5 - GENERACION DE ESPECTROGRAMAS DE SECUENCIAS (reales)")
    print("=" * 60)

    audios = indexar_audios(config.RUTA_AUDIOS)
    print("Audios encontrados:", len(audios))

    df = pd.read_csv(config.ARCHIVO_MAESTRO)
    df = df[df["Tipo_Anotacion"] == "Secuencia"].copy()
    print("Secuencias cargadas del archivo maestro:", len(df))

    grupos = defaultdict(list)
    no_encontrados = 0
    for _, fila in df.iterrows():
        nombre_audio = fila["Begin File"]
        if nombre_audio not in audios:
            no_encontrados += 1
            continue
        grupos[nombre_audio].append(fila.to_dict())
    if no_encontrados:
        print("Audios no encontrados (omitidos):", no_encontrados)

    tareas = [(nombre, audios[nombre], filas) for nombre, filas in grupos.items()]
    print(f"Procesando {len(tareas)} audios con {N_PROCESOS} procesos "
          f"(ventana: {config.DURACION_CHUNK_SECUENCIA:.1f} s)...")

    todos_los_metadatos = []
    todas_las_omitidas = []
    contador_global = defaultdict(int)
    contador_ventanas = defaultdict(int)
    audios_con_nyquist_recortado = {}

    with mp.Pool(processes=N_PROCESOS) as pool:
        for resultados, omitidas in pool.imap_unordered(procesar_audio, tareas):
            todos_los_metadatos.extend(resultados)
            todas_las_omitidas.extend(omitidas)
            for r in resultados:
                contador_global[r["especie"]] += 1
                contador_ventanas[r["ventana_total"]] += 1
                if r["frecuencia_recortada_por_nyquist"]:
                    audios_con_nyquist_recortado[r["audio_origen"]] = (r["sample_rate_hz"], r["high_freq_usado_hz"])

    ruta_metadatos = os.path.join(config.SALIDA_ESPECTROGRAMAS_SECUENCIAS, "metadatos.csv")
    if todos_los_metadatos:
        campos = list(todos_los_metadatos[0].keys())
        with open(ruta_metadatos, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=campos)
            writer.writeheader()
            writer.writerows(todos_los_metadatos)
        print("Metadatos guardados en:", ruta_metadatos)

    if audios_con_nyquist_recortado:
        ruta_nyquist = os.path.join(config.SALIDA_ESPECTROGRAMAS_SECUENCIAS, "audios_con_nyquist_menor_a_esperado.csv")
        pd.DataFrame(
            [{"audio": a, "sample_rate_hz": s, "high_freq_usado_hz": hf} for a, (s, hf) in audios_con_nyquist_recortado.items()]
        ).to_csv(ruta_nyquist, index=False)
        print(f"\n[AVISO] {len(audios_con_nyquist_recortado)} audios con Nyquist menor al HIGH_FREQ configurado: {ruta_nyquist}")

    if todas_las_omitidas:
        ruta_omitidas = os.path.join(config.SALIDA_ESPECTROGRAMAS_SECUENCIAS, "ventanas_omitidas_audio_corto.csv")
        pd.DataFrame(todas_las_omitidas).to_csv(ruta_omitidas, index=False)

        ids_con_imagen = {r["anotacion_id"] for r in todos_los_metadatos}
        ids_con_omision = {o["anotacion_id"] for o in todas_las_omitidas}
        ids_totalmente_perdidos = ids_con_omision - ids_con_imagen

        print(f"\n[AVISO] {len(todas_las_omitidas)} ventanas omitidas por archivo .wav mas corto que "
              f"{config.DURACION_CHUNK_SECUENCIA:.1f}s (no hay audio real suficiente para completarlas).")
        print(f"  De esas, {len(ids_totalmente_perdidos)} secuencias quedaron SIN NINGUNA imagen "
              f"(todas sus ventanas cayeron en audios demasiado cortos).")
        print(f"  Detalle completo: {ruta_omitidas}")
        if ids_totalmente_perdidos:
            ruta_perdidas = os.path.join(config.SALIDA_ESPECTROGRAMAS_SECUENCIAS, "secuencias_sin_ninguna_imagen.csv")
            pd.DataFrame(sorted(ids_totalmente_perdidos), columns=["Anotacion_ID"]).to_csv(ruta_perdidas, index=False)
            print(f"  Lista de esas secuencias (para decidir que hacer con ellas): {ruta_perdidas}")

    print("\n" + "=" * 60)
    print("FASE 5 FINALIZADA")
    print("=" * 60)
    print(f"Imagenes generadas: {len(todos_los_metadatos)}")
    print("\nPor especie:")
    for especie, cantidad in sorted(contador_global.items()):
        print(" ", especie, "->", cantidad)
    print("\nPor numero de ventanas generadas por secuencia:")
    for n_vent, cantidad in sorted(contador_ventanas.items()):
        print(f"  {n_vent} ventana(s) -> {cantidad} imagenes")


if __name__ == "__main__":
    main()

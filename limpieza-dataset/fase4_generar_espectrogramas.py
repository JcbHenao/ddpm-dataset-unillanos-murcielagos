"""
FASE 4 - GENERACION DE ESPECTROGRAMAS (solo pulsos individuales)
=====================================================
Adaptacion de generar_espectrogramas_pulsos_datos_anotados.py: la logica
de generacion de espectrogramas (ventana fija centrada, STFT, recorte en
frecuencia, nombre de archivo) queda IGUAL. Lo unico que cambia es la
fuente: en vez de leer "anotaciones_pulsos_limpias.csv" (que solo tenia
Pulsos), lee el archivo maestro unico (config.ARCHIVO_MAESTRO) y filtra
Tipo_Anotacion == "Pulso" antes de generar. Las filas de tipo Secuencia
quedan fuera de esta fase (son la entrada del modelo de secuencias, un
paso posterior y separado del proyecto).
=====================================================
"""

import os
import gc
import csv
import multiprocessing as mp
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import numpy as np

import librosa
import librosa.display
import matplotlib.pyplot as plt

import config

LOW_FREQ = 20000
HIGH_FREQ = 120000

N_FFT = 512
HOP_LENGTH = 64
DPI = 128

N_PROCESOS = os.cpu_count()


def limpiar_nombre(nombre):
    caracteres = [" ", "/", "\\", ":", "|", "{", "}"]
    for c in caracteres:
        nombre = nombre.replace(c, "_")
    return nombre


def limpiar_o_marcar_na(valor, marcador_na="SinDato"):
    if pd.isna(valor):
        return marcador_na
    return limpiar_nombre(str(valor))


def indexar_audios(ruta):
    indice = {}
    for root, _, files in os.walk(ruta):
        for archivo in files:
            if archivo.lower().endswith(".wav"):
                indice[archivo] = os.path.join(root, archivo)
    return indice


def procesar_audio(args):
    nombre_audio, audio_path, filas = args
    resultados_meta = []
    contador_local = defaultdict(int)

    try:
        audio, sr = librosa.load(audio_path, sr=None, mono=True)
    except Exception as e:
        print("Error cargando:", nombre_audio, e)
        return resultados_meta

    for fila in filas:
        try:
            inicio = float(fila["Begin Time (s)"])
            fin = float(fila["End Time (s)"])
            centro = (inicio + fin) / 2

            inicio_ventana = centro - config.DURACION_OBJETIVO_VENTANA / 2
            n_muestras_objetivo = int(round(config.DURACION_OBJETIVO_VENTANA * sr))

            ventana_centrada = True

            if inicio_ventana < 0:
                inicio_ventana = 0.0
                ventana_centrada = False

            idx_inicio = int(round(inicio_ventana * sr))
            idx_fin = idx_inicio + n_muestras_objetivo

            if idx_fin > len(audio):
                idx_fin = len(audio)
                idx_inicio = max(0, idx_fin - n_muestras_objetivo)
                ventana_centrada = False

            inicio_ventana = idx_inicio / sr
            fin_ventana = idx_fin / sr

            muestra = audio[idx_inicio:idx_fin]

            if len(muestra) < n_muestras_objetivo:
                print(
                    f"Pulso omitido (audio mas corto que la ventana): "
                    f"{nombre_audio}, muestras disponibles={len(muestra)}, "
                    f"requeridas={n_muestras_objetivo}"
                )
                continue

            S = librosa.stft(muestra, n_fft=N_FFT, hop_length=HOP_LENGTH)
            S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
            S_db = np.clip(S_db, -80, 0)

            fig, ax = plt.subplots(figsize=(4, 4))
            librosa.display.specshow(
                S_db, sr=sr, hop_length=HOP_LENGTH,
                x_axis="time", y_axis="hz", cmap="jet", ax=ax,
            )
            ax.set_ylim(LOW_FREQ, HIGH_FREQ)
            ax.axis("off")

            especie = limpiar_nombre(fila["Specie"])
            calidad = limpiar_o_marcar_na(fila["Calidad"])
            pulse_type = limpiar_o_marcar_na(fila["Pulse Type"])
            anotacion_id = fila.get("Anotacion_ID", "")

            contador_local[especie] += 1
            numero = contador_local[especie]

            audio_base = limpiar_nombre(os.path.splitext(nombre_audio)[0])

            nombre_salida = (
                f"{audio_base}"
                f"__t{inicio_ventana:.3f}-{fin_ventana:.3f}s"
                f"__{pulse_type}"
                f"__{calidad}"
                f"__p{numero:05d}.png"
            )

            carpeta = os.path.join(config.SALIDA_ESPECTROGRAMAS, especie)
            os.makedirs(carpeta, exist_ok=True)
            ruta = os.path.join(carpeta, nombre_salida)

            plt.tight_layout(pad=0)
            plt.savefig(ruta, dpi=DPI, bbox_inches="tight", pad_inches=0)
            plt.close(fig)

            resultados_meta.append({
                "archivo_imagen": nombre_salida,
                "especie": especie,
                "audio_origen": nombre_audio,
                "inicio_s": f"{inicio_ventana:.6f}",
                "fin_s": f"{fin_ventana:.6f}",
                "pulse_type": pulse_type,
                "calidad": calidad,
                "anotacion_id": anotacion_id,
                "ventana_centrada": ventana_centrada,
            })

            del muestra, S, S_db

        except Exception as e:
            print("Error procesando pulso de", nombre_audio, ":", e)
            continue

    del audio
    gc.collect()
    return resultados_meta


def main():
    os.makedirs(config.SALIDA_ESPECTROGRAMAS, exist_ok=True)

    print("=" * 60)
    print("FASE 4 - GENERACION DE ESPECTROGRAMAS (solo Pulsos)")
    print("=" * 60)

    audios = indexar_audios(config.RUTA_AUDIOS)
    print("Audios encontrados:", len(audios))

    df = pd.read_csv(config.ARCHIVO_MAESTRO)
    df = df[df["Tipo_Anotacion"] == "Pulso"].copy()
    print("Pulsos cargados del archivo maestro (Secuencias excluidas de esta fase):", len(df))

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
    print(f"Procesando {len(tareas)} audios con {N_PROCESOS} procesos...")

    todos_los_metadatos = []
    contador_global = defaultdict(int)

    with mp.Pool(processes=N_PROCESOS) as pool:
        for resultados in pool.imap_unordered(procesar_audio, tareas):
            todos_los_metadatos.extend(resultados)
            for r in resultados:
                contador_global[r["especie"]] += 1

    ruta_metadatos = os.path.join(config.SALIDA_ESPECTROGRAMAS, "metadatos.csv")
    if todos_los_metadatos:
        campos = list(todos_los_metadatos[0].keys())
        with open(ruta_metadatos, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=campos)
            writer.writeheader()
            writer.writerows(todos_los_metadatos)
        print("Metadatos guardados en:", ruta_metadatos)

    print("\n" + "=" * 60)
    print("FASE 4 FINALIZADA")
    print("=" * 60)
    for especie, cantidad in sorted(contador_global.items()):
        print(especie, "->", cantidad)


if __name__ == "__main__":
    main()

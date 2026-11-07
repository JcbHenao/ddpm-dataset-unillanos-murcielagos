import os
import gc

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import numpy as np

import librosa

from scipy.signal import butter, lfilter

from PIL import Image



# =====================================================
# Configuracion
# =====================================================

RUTA_ANOTACIONES = (
    "//media/adalab/bodega/Andres Muñoz/dinamica-temporal-de-la-actividad-de-murcielagos/Data/"
    "SongMeter_MiniBat/DeteccionPulsos/Evaluacion_ModelosPreentrenados/AnotacionesPorAudio"
)


RUTA_AUDIOS = (
    "/media/adalab/bodega/CamiloBuitrago/audios"
)


SALIDA = (
    "/media/adalab/bodega/CamiloBuitrago/"
    "Espectrogramas_Pulsos"
)



DURACION_OBJETIVO = 0.015


LOW_FREQ = 20000
HIGH_FREQ = 120000


N_FFT = 1024
HOP_LENGTH = 256


IMG_SIZE = (
    512,
    512
)



os.makedirs(
    SALIDA,
    exist_ok=True
)



# =====================================================
# Indexar audios
# =====================================================

def indexar_audios(ruta):

    indice = {}

    for root, _, files in os.walk(ruta):

        for archivo in files:

            if archivo.lower().endswith(".wav"):

                indice[archivo] = os.path.join(
                    root,
                    archivo
                )

    return indice

def indexar_anotaciones(ruta):

    indice = {}

    for root, _, files in os.walk(ruta):

        for archivo in files:

            if archivo.lower().endswith(".xlsx"):

                indice[archivo] = os.path.join(
                    root,
                    archivo
                )

    return indice

audios = indexar_audios(
    RUTA_AUDIOS
)

anotaciones= indexar_anotaciones(RUTA_ANOTACIONES)

print("Anotaciones encontradas:", len(anotaciones))

dfs = []

for archivo, ruta_excel in anotaciones.items():

    try:
        df_temp = pd.read_excel(ruta_excel)

        # opcional: guardar el nombre del archivo origen
        df_temp["archivo_anotacion"] = archivo

        dfs.append(df_temp)

        print(f"OK: {archivo}")

    except Exception as e:

        print(f"ERROR: {archivo} -> {e}")

if len(dfs) == 0:
    raise Exception("No se pudo leer ningún archivo de anotaciones")

df = pd.concat(
    dfs,
    ignore_index=True
)



print("Total registros:", len(df))

print(
    "Audios encontrados:",
    len(audios),
)



# =====================================================
# Utilidades
# =====================================================

def limpiar_nombre(nombre):

    for c in [
        " ",
        "/",
        "\\",
        ":",
        "|"
    ]:

        nombre = nombre.replace(
            c,
            "_"
        )

    return nombre



def filtro_pasabanda(
    audio,
    sr
):

    nyquist = sr / 2


    high = min(
        HIGH_FREQ,
        nyquist * 0.95
    )


    low = LOW_FREQ / nyquist
    high = high / nyquist


    b, a = butter(
        6,
        [
            low,
            high
        ],
        btype="band"
    )


    return lfilter(
        b,
        a,
        audio
    )



def normalizar(S):

    minimo = S.min()
    maximo = S.max()


    if maximo == minimo:

        return np.zeros_like(
            S
        )


    return (
        S - minimo
    ) / (
        maximo - minimo
    )



def generar_espectrograma(
    audio,
    sr
):

    audio = filtro_pasabanda(
        audio,
        sr
    )


    S = librosa.stft(
        audio,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )


    S_db = librosa.amplitude_to_db(
        np.abs(S),
        ref=np.max
    )


    S_norm = normalizar(
        S_db
    )


    imagen = (
        S_norm * 255
    ).astype(
        np.uint8
    )


    imagen = np.flipud(
        imagen
    )


    return imagen



def generar_ventanas(
    inicio,
    fin,
    dur_audio
):

    duracion = fin - inicio


    ventanas = []



    # -----------------------------
    # Evento menor o igual
    # -----------------------------

    if duracion <= DURACION_OBJETIVO:

        centro = (
            inicio + fin
        ) / 2


        inicio_v = (
            centro -
            DURACION_OBJETIVO / 2
        )


        fin_v = (
            inicio_v +
            DURACION_OBJETIVO
        )


        if inicio_v < 0:

            inicio_v = 0
            fin_v = DURACION_OBJETIVO


        if fin_v > dur_audio:

            fin_v = dur_audio
            inicio_v = (
                dur_audio -
                DURACION_OBJETIVO
            )


        ventanas.append(
            (
                inicio_v,
                fin_v
            )
        )


    # -----------------------------
    # Evento largo
    # -----------------------------

    else:

        actual = inicio


        while (
            actual + DURACION_OBJETIVO
            <= fin
        ):

            ventanas.append(
                (
                    actual,
                    actual + DURACION_OBJETIVO
                )
            )


            actual += DURACION_OBJETIVO



        if actual < fin:

            ventanas.append(
                (
                    fin - DURACION_OBJETIVO,
                    fin
                )
            )



    return ventanas



# =====================================================
# Cargar anotaciones
# =====================================================



contador_global = 1
contador_anotacion = 1



# =====================================================
# Procesamiento
# =====================================================

for _, fila in df.iterrows():


    nombre_audio = fila[
        "Begin File"
    ]


    if nombre_audio not in audios:

        continue



    audio_path = audios[
        nombre_audio
    ]



    try:

        audio, sr = librosa.load(
            audio_path,
            sr=None
        )


    except:

        continue



    dur_audio = len(audio) / sr



    ventanas = generar_ventanas(
        fila["Begin Time (s)"],
        fila["End Time (s)"],
        dur_audio
    )


    total_seq = len(
        ventanas
    )



    especie = limpiar_nombre(
        str(
            fila["Specie"]
        )
    )


    calidad = limpiar_nombre(
        str(
            fila["Calidad"]
        )
    )



    pulse = limpiar_nombre(
        str(
            fila.get(
                "Pulse Type",
                "NA"
            )
        )
    )



    audio_base = limpiar_nombre(
        os.path.splitext(
            nombre_audio
        )[0]
    )



    carpeta = os.path.join(
        SALIDA,
        especie
    )


    os.makedirs(
        carpeta,
        exist_ok=True
    )



    for numero_seq, (ini, fin) in enumerate(
        ventanas,
        start=1
    ):


        muestra = audio[
            int(ini*sr):
            int(fin*sr)
        ]



        espectro = generar_espectrograma(
            muestra,
            sr
        )


        imagen = Image.fromarray(
            espectro,
            mode="L"
        )


        imagen = imagen.resize(
            IMG_SIZE,
            Image.LANCZOS
        )



        nombre = (

            f"{audio_base}"

            f"__seq{numero_seq:02d}"
            f"of{total_seq:02d}"

            f"__{ini:.3f}-{fin:.3f}s"

            f"_{especie}"

            f"_{pulse}.png"

        )



        salida = os.path.join(
            carpeta,
            nombre
        )


        imagen.save(
            salida
        )


        contador_global += 1



    contador_anotacion += 1


    del audio
    gc.collect()



print("==============================")
print("GENERACION FINALIZADA")
print("==============================")


print(
    "Total espectrogramas:",
    contador_global - 1
)

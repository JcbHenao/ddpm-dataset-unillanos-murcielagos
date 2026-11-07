import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt


# =====================================================
# Configuracion de rutas
# =====================================================

RUTA_ANOTACIONES = (
    "/media/adalab/bodega/CamiloBuitrago/Generacion_Espectrogramas/"
    "anotaciones_normalizadas.csv"
)


RUTA_AUDIOS = (
    "/media/adalab/bodega/CamiloBuitrago/audios"
)


SALIDA = (
    "/media/adalab/bodega/CamiloBuitrago/"
    "Espectrogramas_Secuencias"
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



audios = indexar_audios(
    RUTA_AUDIOS
)


print(
    "Audios encontrados:",
    len(audios)
)



# =====================================================
# Limpieza nombres
# =====================================================

def limpiar_nombre(nombre):

    caracteres = [
        " ",
        "/",
        "\\",
        ":",
        "|"
    ]

    for c in caracteres:

        nombre = nombre.replace(
            c,
            "_"
        )

    return nombre



# =====================================================
# Cargar anotaciones
# =====================================================

df = pd.read_csv(
    RUTA_ANOTACIONES
)


print(
    "Anotaciones:",
    len(df)
)



# =====================================================
# Agrupar por audio
# =====================================================

grupos = df.groupby(
    "Begin File"
)



# =====================================================
# Contadores por especie
# =====================================================

contador_especies = defaultdict(int)



# =====================================================
# Procesamiento
# =====================================================

for nombre_audio, grupo in grupos:


    if nombre_audio not in audios:

        print(
            "Audio no encontrado:",
            nombre_audio
        )

        continue



    audio_path = audios[nombre_audio]


    try:

        y, sr = librosa.load(
            audio_path,
            sr=None
        )


    except Exception as e:

        print(
            "Error cargando:",
            nombre_audio,
            e
        )

        continue



    # -------------------------------------------------
    # Secuencia
    # -------------------------------------------------

    secuencia_id = 1



    t_min = grupo[
        "Begin Time (s)"
    ].min()


    t_max = grupo[
        "End Time (s)"
    ].max()



    # contexto temporal

    t_ini = max(
        0,
        t_min - 1
    )


    t_fin = min(
        len(y)/sr,
        t_max + 1
    )



    y_cut = y[
        int(t_ini*sr):
        int(t_fin*sr)
    ]



    # -------------------------------------------------
    # STFT
    # -------------------------------------------------

    S = librosa.stft(
        y_cut,
        n_fft=1024,
        hop_length=256
    )


    S_db = librosa.amplitude_to_db(
        np.abs(S),
        ref=np.max
    )



    fig, ax = plt.subplots(
        figsize=(10,5)
    )



    librosa.display.specshow(

        S_db,

        sr=sr,

        hop_length=256,

        x_axis="time",

        y_axis="hz",

        cmap="jet",

        ax=ax

    )


    ax.set_ylim(
        0,
        100000
    )



    # -------------------------------------------------
    # Bounding boxes
    # -------------------------------------------------

    for _, fila in grupo.iterrows():


        inicio = (
            fila["Begin Time (s)"]
            -
            t_ini
        )


        fin = (
            fila["End Time (s)"]
            -
            t_ini
        )


        frecuencia_baja = max(
            0,
            fila["Low Freq (Hz)"]
        )


        frecuencia_alta = min(
            100000,
            fila["High Freq (Hz)"]
        )



        rectangulo = plt.Rectangle(

            (
                inicio,
                frecuencia_baja
            ),

            fin - inicio,

            frecuencia_alta - frecuencia_baja,

            fill=False,

            edgecolor="cyan",

            linewidth=1

        )


        ax.add_patch(
            rectangulo
        )



    # -------------------------------------------------
    # Metadatos
    # -------------------------------------------------

    especie = limpiar_nombre(
        str(
            grupo["Specie"]
            .iloc[0]
        )
    )


    calidad = limpiar_nombre(
        str(
            grupo["Calidad"]
            .iloc[0]
        )
    )



    if "Pulse Type" in grupo.columns:

        pulse_type = limpiar_nombre(
            str(
                grupo["Pulse Type"]
                .iloc[0]
            )
        )

    else:

        pulse_type = "NA"



    audio_base = limpiar_nombre(
        os.path.splitext(
            nombre_audio
        )[0]
    )



    # contador independiente

    contador_especies[especie] += 1


    numero_especie = contador_especies[especie]



    duracion_sec = (
        t_max - t_min
    )



    # -------------------------------------------------
    # Titulo visual
    # -------------------------------------------------

    ax.set_title(

        f"Especie: {especie}\n"
        f"Calidad: {calidad} | "
        f"Pulse Type: {pulse_type}\n"
        f"Audio: {nombre_audio}\n"
        f"Secuencia: {secuencia_id} | "
        f"{t_min:.3f}s - {t_max:.3f}s | "
        f"Duracion: {duracion_sec:.3f}s\n"
        f"Muestra especie: {numero_especie:05d}",

        fontsize=9

    )



    # -------------------------------------------------
    # Nombre archivo
    # -------------------------------------------------

    nombre_salida = (

        f"{audio_base}"

        f"_seq_{secuencia_id:04d}"

        f"__{t_min:.3f}-{t_max:.3f}s"

        f"_{especie}"

        f"_{calidad}"

        f"_{pulse_type}"

        f"__{numero_especie:05d}.png"

    )



    # -------------------------------------------------
    # Carpeta especie
    # -------------------------------------------------

    carpeta_especie = os.path.join(
        SALIDA,
        especie
    )


    os.makedirs(
        carpeta_especie,
        exist_ok=True
    )



    ruta_guardado = os.path.join(
        carpeta_especie,
        nombre_salida
    )



    # -------------------------------------------------
    # Guardar
    # -------------------------------------------------

    plt.tight_layout()


    plt.savefig(
        ruta_guardado,
        dpi=120,
        bbox_inches="tight"
    )


    plt.close(fig)



    print(
        "Generado:",
        ruta_guardado
    )



print("\n==============================")
print("PROCESO TERMINADO")
print("==============================")


print("\nDistribucion por especie:")

for especie, cantidad in sorted(
    contador_especies.items()
):

    print(
        f"{especie:25s} -> {cantidad}"
    )

import numpy as np
import matplotlib.cm as cm
from PIL import Image
import librosa

LOW_FREQ = 20000
HIGH_FREQ = 100000

N_FFT = 512
ANCHO_IMG_PX = 2048  
ALTO_IMG_PX = 128    
RANGO_DB = 80.0  # rango dinamico fijo: -80..0 dB para TODAS las imagenes


def limpiar_nombre(nombre):
    caracteres = [" ", "/", "\\", ":", "|", "{", "}"]
    for c in caracteres:
        nombre = nombre.replace(c, "_")
    return nombre


def limpiar_o_marcar_na(valor, marcador_na="SinDato"):
    import pandas as pd
    if pd.isna(valor):
        return marcador_na
    return limpiar_nombre(str(valor))


def indexar_audios(ruta):
    import os
    indice = {}
    for root, _, files in os.walk(ruta):
        for archivo in files:
            if archivo.lower().endswith(".wav"):
                indice[archivo] = os.path.join(root, archivo)
    return indice

def generar_ventanas(inicio_evento, fin_evento, duracion_ventana):

    duracion_evento = fin_evento - inicio_evento
    ventanas = []

    # ---------------------------------------------------------
    # La secuencia es menor que una ventana
    # ---------------------------------------------------------
    if duracion_evento < duracion_ventana:
        centro_evento = (inicio_evento + fin_evento) / 2

        inicio_ventana = (
            centro_evento - duracion_ventana / 2
        )

        fin_ventana = (
            inicio_ventana + duracion_ventana
        )

        ventanas.append(
            (inicio_ventana, fin_ventana)
        )

        return ventanas

    # ---------------------------------------------------------
    # Ventanas consecutivas desde el inicio de la secuencia
    # ---------------------------------------------------------
    inicio = inicio_evento

    while inicio + duracion_ventana <= fin_evento + 1e-6:

        ventanas.append(
            (
                inicio,
                inicio + duracion_ventana
            )
        )

        inicio += duracion_ventana

    # ---------------------------------------------------------
    # Ajustar ÚNICAMENTE la última parte de la secuencia
    # ---------------------------------------------------------
    if inicio < fin_evento - 1e-6:

        inicio_ultima = fin_evento - duracion_ventana

        # Evitar duplicar la ventana anterior
        if inicio_ultima > ventanas[-1][0] + 1e-6:

            ventanas.append(
                (
                    inicio_ultima,
                    fin_evento
                )
            )

    return ventanas


def extraer_ventana(audio, sr, inicio_ventana, duracion_ventana):
    """
    Exactamente n_muestras_objetivo muestras de audio REAL. Si la
    ventana pedida se sale del archivo, se recorre (shift) para caer
    dentro -- nunca se rellena con ceros. Si el archivo completo es mas
    corto que la ventana, devuelve lo que haya y marca
    audio_insuficiente=True para que el llamador la descarte (no hay
    audio real suficiente para fabricar).
    """
    n_muestras_objetivo = int(round(duracion_ventana * sr))
    idx_inicio = int(round(inicio_ventana * sr))
    idx_fin = idx_inicio + n_muestras_objetivo
    longitud_audio = len(audio)

    ventana_ajustada_borde = False

    if idx_inicio < 0:
        idx_inicio = 0
        idx_fin = idx_inicio + n_muestras_objetivo
        ventana_ajustada_borde = True

    if idx_fin > longitud_audio:
        idx_fin = longitud_audio
        idx_inicio = max(0, idx_fin - n_muestras_objetivo)
        ventana_ajustada_borde = True

    muestra = audio[idx_inicio:idx_fin]
    audio_insuficiente = len(muestra) < n_muestras_objetivo

    inicio_real = idx_inicio / sr
    fin_real = (idx_inicio + len(muestra)) / sr
    return muestra, inicio_real, fin_real, ventana_ajustada_borde, audio_insuficiente


def calcular_hop_length(duracion_ventana, sr):
    n_muestras_ventana = int(round(duracion_ventana * sr))
    return max(1, n_muestras_ventana // (ANCHO_IMG_PX * SOBREMUESTREO_FRAMES)), n_muestras_ventana


def render_espectrograma_uniforme(muestra, sr, hop_length, ruta_salida):
    """
    STFT -> dB -> recorte a banda de frecuencia real (Nyquist-safe) ->
    normalizacion a rango dB FIJO (no por imagen) -> colormap jet ->
    PNG de tamaño EXACTO (ANCHO_IMG_PX x ALTO_IMG_PX) via PIL.
    Devuelve el high_freq real usado (== HIGH_FREQ si el archivo si
    tiene esa frecuencia, o el Nyquist si no).
    """
    S = librosa.stft(muestra, n_fft=N_FFT, hop_length=hop_length)
    S_db = np.clip(librosa.amplitude_to_db(np.abs(S), ref=np.max), -RANGO_DB, 0)

    n_bins = S_db.shape[0]  # n_fft // 2 + 1
    nyquist = sr / 2
    freq_por_bin = nyquist / (n_bins - 1)

    high_freq_real = min(HIGH_FREQ, nyquist)
    bin_low = max(0, int(np.floor(LOW_FREQ / freq_por_bin)))
    bin_high = min(n_bins - 1, int(np.ceil(high_freq_real / freq_por_bin)))

    recorte = S_db[bin_low:bin_high + 1, :]
    recorte_norm = np.clip((recorte + RANGO_DB) / RANGO_DB, 0, 1)  # rango FIJO -80..0, no min/max de la imagen
    recorte_norm = recorte_norm[::-1, :]  # frecuencia alta arriba

    rgba = cm.jet(recorte_norm)
    rgb = (rgba[:, :, :3] * 255).astype(np.uint8)

    Image.fromarray(rgb).resize((ANCHO_IMG_PX, ALTO_IMG_PX), Image.LANCZOS).save(ruta_salida)
    return high_freq_real

"""
Render comun de las pseudo-secuencias.

Regla: un pixel significa lo mismo en TODAS las imagenes.
  - Tiempo: todos los audios se remuestrean a sr_comun y se usa un hop fijo,
    asi que cada columna mide los mismos milisegundos y no se redimensiona nada.
  - Frecuencia: con sr_comun / n_fft = 625 Hz por fila, 20-100 kHz son
    exactamente 128 filas. Audios con Nyquist menor quedan con la parte alta
    en el piso (no se estira).
  - Intensidad: IGUAL que render_espectrograma_uniforme del pipeline de los
    experimentos previos (espectrogramas_secuencias_comun.py):
      S_db = amplitude_to_db(|STFT|, ref=np.max) sobre TODO el espectro,
      recorte fijo a [-80, 0] dB, luego recorte de la banda 20-100 kHz,
      (S_db + 80) / 80 -> colormap jet -> uint8 (truncado, como cm.jet(...)*255).
    No se aplica ninguna normalizacion ni filtro adicional.
  - La ventana es audio real desde margen antes del primer pulso hasta margen
    despues del ultimo (el silencio entre pulsos se conserva). Las columnas
    que sobran a la derecha quedan en el piso (-rango_db).
"""

import math

import librosa
import matplotlib
import numpy as np
from PIL import Image


def geometria(p_render, banda_hz):
    sr, n_fft = p_render["sr_comun"], p_render["n_fft"]
    hz_fila = sr / n_fft
    b0, b1 = banda_hz[0] / hz_fila, banda_hz[1] / hz_fila
    if abs(b0 - round(b0)) > 1e-9 or abs(b1 - round(b1)) > 1e-9:
        raise ValueError(f"La banda {banda_hz} no cae en filas exactas con sr_comun={sr} y n_fft={n_fft} "
                         f"({hz_fila} Hz por fila). Ajustar sr_comun o n_fft en parametros.yaml.")
    return {"sr": sr, "n_fft": n_fft, "hz_por_fila": hz_fila,
            "bin_bajo": int(round(b0)), "bin_alto": int(round(b1)), "alto_px": int(round(b1 - b0))}


def hop_para(duracion_ms, ancho_px, sr, n_fft):
    """Hop (muestras) minimo para que ancho_px columnas cubran duracion_ms."""
    muestras = duracion_ms / 1000 * sr
    return max(1, math.ceil((muestras - n_fft) / (ancho_px - 1)))


def duracion_efectiva_ms(hop, ancho_px, sr, n_fft):
    return (n_fft + (ancho_px - 1) * hop) / sr * 1000


def extraer(audio, sr_orig, t0, t1, sr_dest, contexto_s=0.002):
    """Audio real [t0, t1] remuestreado a sr_dest; usa un poco de contexto para evitar efectos de borde."""
    a = max(0, int(round((t0 - contexto_s) * sr_orig)))
    b = min(len(audio), int(round((t1 + contexto_s) * sr_orig)))
    seg = audio[a:b].astype(np.float32)
    if sr_orig != sr_dest:
        seg = librosa.resample(seg, orig_sr=sr_orig, target_sr=sr_dest, res_type="soxr_hq")
    inicio = int(round((t0 - a / sr_orig) * sr_dest))
    largo = int(round((t1 - t0) * sr_dest))
    return seg[inicio:inicio + largo]


def espectrograma_db(x, geo, hop, ancho_px, rango_db):
    """
    Matriz (alto_px, ancho_px) en dB [-rango_db, 0], frecuencia alta arriba, con la
    misma normalizacion que render_espectrograma_uniforme del pipeline previo:
    STFT con los valores por defecto de librosa (center=True, ventana hann),
    referencia = maximo de TODO el espectro (no solo de la banda), recorte a
    [-rango_db, 0]. Devuelve tambien cuantas columnas tienen audio.
    Las columnas sin audio (a la derecha) quedan en -rango_db, el mismo valor
    que el recorte inferior.
    """
    n_fft = geo["n_fft"]
    lienzo = np.full((geo["alto_px"], ancho_px), -float(rango_db), dtype=np.float32)
    if len(x) < n_fft:
        return lienzo[::-1].copy(), 0
    S = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop))
    S_db = np.clip(librosa.amplitude_to_db(S, ref=np.max), -rango_db, 0)
    S_db = S_db[geo["bin_bajo"]:geo["bin_alto"], :ancho_px]
    columnas = S_db.shape[1]
    lienzo[:, :columnas] = S_db
    return lienzo[::-1].copy(), columnas


def a_imagen(db, rango_db, colormap="jet"):
    norm = np.clip((db + rango_db) / rango_db, 0, 1)
    if colormap == "gris":
        return Image.fromarray((norm * 255).round().astype(np.uint8))
    rgb = (matplotlib.colormaps[colormap](norm)[:, :, :3] * 255).astype(np.uint8)   # truncado, como el pipeline
    return Image.fromarray(rgb)


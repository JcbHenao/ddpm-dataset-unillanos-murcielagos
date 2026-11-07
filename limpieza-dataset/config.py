"""
config.py
=====================================================
Configuracion centralizada de rutas para el pipeline de anotaciones.

Antes (bola de scripts): cada script (normalizar_anotaciones.py,
limpiar_anotaciones_pulsos.py, audit_*.py, generar_espectrogramas*.py)
repetia sus propias rutas hardcodeadas. Cambiar una ruta significaba
tocar 8 archivos distintos, y era facil que quedaran desincronizados
(de hecho asi paso: limpiar_anotaciones_pulsos.py quedo como una
version vieja de normalizar_anotaciones.py con las mismas rutas pero
logica desactualizada).

Ahora: un solo lugar. Todas las fases (fase0..fase4) importan de aqui.

Para correr el pipeline en otra maquina o en local para pruebas, se
puede sobreescribir la ruta base con la variable de entorno
ANOTACIONES_BASE, sin tocar este archivo.
=====================================================
"""

import os

from espectrogramas_secuencias_comun import ANCHO_IMG_PX as _ANCHO_SECUENCIAS
from espectrogramas_secuencias_comun import ALTO_IMG_PX as _ALTO_SECUENCIAS

# =====================================================
# Ruta base (todo lo demas cuelga de aqui)
# =====================================================

RUTA_BASE = os.environ.get(
    "ANOTACIONES_BASE",
    "/media/adalab/bodega/CamiloBuitrago",
)


def resolver(base, *partes):
    """
    Resuelve base/partes[0]/partes[1]/... nivel por nivel, sin importar
    mayusculas/minusculas: si el nombre exacto no existe en el
    filesystem pero hay una carpeta con el mismo nombre en otra
    capitalizacion (ej. se pide "Anotaciones" y en el servidor la
    carpeta real es "anotaciones"), usa esa. Evita que un simple
    mismatch de mayusculas (Linux distingue mayusculas de minusculas)
    haga que una fase encuentre "0 archivos" en silencio en vez de dar
    un error claro. Si no encuentra ninguna coincidencia (ni exacta ni
    insensible a mayusculas), devuelve la ruta tal como se pidio, para
    que el mensaje de "no existe" aguas abajo siga siendo claro sobre
    que se esperaba.
    """
    actual = base
    for parte in partes:
        candidato_exacto = os.path.join(actual, parte)
        if os.path.exists(candidato_exacto):
            actual = candidato_exacto
            continue
        try:
            entradas = os.listdir(actual)
        except OSError:
            entradas = []
        coincidencia = next((e for e in entradas if e.lower() == parte.lower()), None)
        actual = os.path.join(actual, coincidencia if coincidencia else parte)
    return actual

# =====================================================
# FUENTES CRUDAS (fase 0 y fase 1 y fase 2 leen de aqui)
# =====================================================


RUTA_ANOTACIONES = resolver(RUTA_BASE, "anotaciones", "Anotaciones")
RUTA_PULSOS = resolver(RUTA_ANOTACIONES, "Pulsos")
RUTA_SECUENCIAS = resolver(RUTA_ANOTACIONES, "Secuencias")

RUTA_TOTAL_ANOTACIONES = resolver(RUTA_BASE, "anotaciones", "Total_Anotaciones")
RUTA_SECUENCIAS_POR_AUDIO = resolver(RUTA_BASE, "anotaciones", "anotaciones_secuencias_por_audio")

RUTA_SALIDAS_MODELO_DETECCION = resolver(RUTA_BASE, "anotaciones", "Modelo_FasterRCNN_Saccopteryx_Molossus")

RUTA_AUDIOS = resolver(RUTA_BASE, "audios")

# =====================================================
# Diccionarios de mapeo (typo -> categoria final)
# =====================================================

# servidor (scripts, mapeos, y las salidas que genera cada fase)
RUTA_GENERACION_ESPECTROGRAMAS = resolver(RUTA_BASE, "Generacion_Espectrogramas")

# Mapeos: viven en una subcarpeta "mapeos/" dentro de Generacion_Espectrogramas
RUTA_MAPEOS = resolver(RUTA_GENERACION_ESPECTROGRAMAS, "mapeos")

RUTA_MAPEO_ESPECIES = os.path.join(RUTA_MAPEOS, "mapeo_especies.csv")
RUTA_MAPEO_CALIDAD = os.path.join(RUTA_MAPEOS, "mapeo_calidad.csv")
RUTA_MAPEO_PULSE_TYPE = os.path.join(RUTA_MAPEOS, "mapeo_pulse_type.csv")

# Fusion final de Calidad: de 5 categorias normalizadas a 3 de uso
FUSION_CALIDAD_FINAL = {
    "Baja": "Baja",
    "Alta": "Alta",
    "Buena": "Alta",
    "Regular": "Media",
    "Media": "Media",
}

# =====================================================
# Salidas del pipeline (todas cuelgan de una sola carpeta por fase,
# para que sea facil ubicar "que genero que")
# =====================================================

SALIDA_BASE = os.path.join(RUTA_GENERACION_ESPECTROGRAMAS, "Anotaciones_Unificadas")

SALIDA_FASE0 = os.path.join(SALIDA_BASE, "Fase0_Inventario")
SALIDA_FASE1 = os.path.join(SALIDA_BASE, "Fase1_Auditoria_Categorias_Crudas")
SALIDA_FASE2 = os.path.join(SALIDA_BASE, "Fase2_Normalizado")
SALIDA_FASE3 = os.path.join(SALIDA_BASE, "Fase3_Auditoria_Resultado")

# Fase 2: el archivo unico que unifica Pulsos + Secuencias, deduplicado
ARCHIVO_MAESTRO = os.path.join(SALIDA_FASE2, "anotaciones_unificadas.csv")
SALIDA_FASE2_POR_ESPECIE = os.path.join(SALIDA_FASE2, "Por_Especie")

# Fase 4: espectrogramas de PULSOS (consume ARCHIVO_MAESTRO, filtra
# Tipo_Anotacion=="Pulso")
SALIDA_ESPECTROGRAMAS = os.path.join(
    RUTA_GENERACION_ESPECTROGRAMAS, "Espectrogramas_Pulsos_Anotados_512px"
)

# Debe coincidir entre fase3 (auditoria) y fase4 (generacion)
DURACION_OBJETIVO_VENTANA = 0.020  # 20 ms

# Fase 5: espectrogramas de SECUENCIAS

DURACION_CHUNK_SECUENCIA = 1.5  
SALIDA_ESPECTROGRAMAS_SECUENCIAS = os.path.join(
    RUTA_GENERACION_ESPECTROGRAMAS,
    f"Espectrogramas_Secuencias_Chunks_{DURACION_CHUNK_SECUENCIA}s_{_ANCHO_SECUENCIAS}x{_ALTO_SECUENCIAS}px",
)

# Fase 6: espectrogramas de las SECUENCIAS DETECTADAS POR EL MODELO
# (Modelo_FasterRCNN_Saccopteryx_Molossus/**/*.xlsx), para comparar
# visualmente "real" (fase5) vs "modelo" (fase6). Usa exactamente la
# misma ventana y el mismo renderizado que fase5 (ver
# espectrogramas_secuencias_comun.py) para que la comparacion sea
# justa -- carpeta de salida separada para no mezclarlas.
SALIDA_ESPECTROGRAMAS_MODELO = os.path.join(
    RUTA_GENERACION_ESPECTROGRAMAS,
    f"Espectrogramas_Secuencias_Modelo_Chunks_{DURACION_CHUNK_SECUENCIA}s_{_ANCHO_SECUENCIAS}x{_ALTO_SECUENCIAS}px",
)


def crear_carpetas_salida():
    for ruta in [
        SALIDA_BASE,
        SALIDA_FASE0,
        SALIDA_FASE1,
        SALIDA_FASE2,
        SALIDA_FASE2_POR_ESPECIE,
        SALIDA_FASE3,
    ]:
        os.makedirs(ruta, exist_ok=True)
import shutil
from pathlib import Path


# ============================================================
# RUTAS
# ============================================================

RUTA_ORIGEN = Path(
    "/media/adalab/bodega/CamiloBuitrago/"
    "Generacion_Espectrogramas/"
    "Espectrogramas_Secuencias_Chunks_1.5s_2048x128px"
)

RUTA_DESTINO = Path(
    "/media/adalab/bodega/CamiloBuitrago/"
    "espectrogramas/"
    "Espectrogramas_Secuencias_Chunks_1.5s_2048x128px"
)


# ============================================================
# VALIDAR ORIGEN
# ============================================================

if not RUTA_ORIGEN.exists():
    raise FileNotFoundError(
        f"No existe la ruta de origen:\n{RUTA_ORIGEN}"
    )


if not RUTA_ORIGEN.is_dir():
    raise NotADirectoryError(
        f"La ruta de origen no es un directorio:\n{RUTA_ORIGEN}"
    )


# ============================================================
# CREAR DESTINO
# ============================================================

RUTA_DESTINO.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# ENCONTRAR CARPETAS QUE COMIENZAN POR "S"
# ============================================================

carpetas_S = sorted(
    [
        carpeta
        for carpeta in RUTA_ORIGEN.iterdir()
        if carpeta.is_dir()
        and carpeta.name.startswith("S")
    ],
    key=lambda x: x.name
)


# ============================================================
# MOSTRAR CARPETAS ENCONTRADAS
# ============================================================

print("=" * 70)
print("CARPETAS ENCONTRADAS")
print("=" * 70)

if not carpetas_S:
    raise RuntimeError(
        "No se encontraron carpetas cuyo nombre comience por 'S'."
    )


for carpeta in carpetas_S:
    print(f"  - {carpeta.name}")


print()
print(f"Total de carpetas encontradas: {len(carpetas_S)}")


# ============================================================
# COPIAR
# ============================================================

print()
print("=" * 70)
print("INICIANDO COPIA")
print("=" * 70)

for carpeta_origen in carpetas_S:

    carpeta_destino = (
        RUTA_DESTINO / carpeta_origen.name
    )

    print()
    print(f"Copiando: {carpeta_origen.name}")
    print(f"Origen : {carpeta_origen}")
    print(f"Destino: {carpeta_destino}")

    shutil.copytree(
        carpeta_origen,
        carpeta_destino,
        dirs_exist_ok=True
    )

    print("OK")


# ============================================================
# RESUMEN
# ============================================================

print()
print("=" * 70)
print("COPIA FINALIZADA")
print("=" * 70)

print(
    f"Carpetas copiadas: {len(carpetas_S)}"
)

print(
    f"Destino:\n{RUTA_DESTINO}"
)
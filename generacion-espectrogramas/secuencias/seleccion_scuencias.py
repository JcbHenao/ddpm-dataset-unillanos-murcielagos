import random
import shutil
from pathlib import Path

from PIL import Image


# ============================================================
# CONFIGURACIÓN
# ============================================================

RUTA_DATASET = Path(
    "/media/adalab/bodega/CamiloBuitrago/"
    "espectrogramas/"
    "Espectrogramas_Secuencias_Chunks_1.5s_2048x128px"
)

RUTA_SALIDA = (
    RUTA_DATASET /
    "Dataset_DDPM_2048x128px"
)


CLASES = [
    "SBilineata",
    "SLeptura",
    "Saccopteryx_sp",
]


# Cantidad exacta para entrenamiento
CANTIDAD_TRAIN = {
    "SBilineata": 333,
    "SLeptura": 333,
    "Saccopteryx_sp": 334,
}


SEED = 42


ANCHO_ESPERADO = 2048
ALTO_ESPERADO = 128


# ============================================================
# EXTENSIONES VÁLIDAS
# ============================================================

EXTENSIONES = {
    ".png",
    ".jpg",
    ".jpeg",
}


# ============================================================
# FUNCIONES
# ============================================================

def obtener_imagenes(carpeta):
    """
    Busca imágenes recursivamente dentro de una carpeta.
    """

    return sorted(
        [
            archivo
            for archivo in carpeta.rglob("*")
            if archivo.is_file()
            and archivo.suffix.lower()
            in EXTENSIONES
        ]
    )


def validar_imagen(ruta):
    """
    Verifica que la imagen tenga exactamente 2048x128 px.
    """

    try:

        with Image.open(ruta) as imagen:

            ancho, alto = imagen.size

        if (
            ancho != ANCHO_ESPERADO
            or alto != ALTO_ESPERADO
        ):

            return False, (ancho, alto)

        return True, (ancho, alto)

    except Exception as error:

        return False, str(error)


def copiar_imagenes(
    imagenes,
    carpeta_origen,
    carpeta_destino
):
    """
    Copia preservando la estructura relativa
    existente dentro de la carpeta de clase.
    """

    for imagen in imagenes:

        relativa = imagen.relative_to(
            carpeta_origen
        )

        destino = (
            carpeta_destino /
            relativa
        )

        destino.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        shutil.copy2(
            imagen,
            destino
        )


# ============================================================
# VALIDAR DIRECTORIO PRINCIPAL
# ============================================================

if not RUTA_DATASET.exists():

    raise FileNotFoundError(
        f"No existe el dataset:\n{RUTA_DATASET}"
    )


# ============================================================
# CREAR DIRECTORIO DE SALIDA
# ============================================================

RUTA_SALIDA.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SEMILLA
# ============================================================

rng = random.Random(SEED)


# ============================================================
# PROCESAMIENTO
# ============================================================

resumen = []


print()
print("=" * 80)
print("SELECCIÓN DEL DATASET DDPM")
print("=" * 80)


for clase in CLASES:

    carpeta_clase = (
        RUTA_DATASET /
        clase
    )

    if not carpeta_clase.exists():

        raise FileNotFoundError(
            f"No existe la carpeta de clase:\n"
            f"{carpeta_clase}"
        )


    print()
    print("-" * 80)
    print(f"CLASE: {clase}")
    print("-" * 80)


    # --------------------------------------------------------
    # Buscar imágenes
    # --------------------------------------------------------

    imagenes = obtener_imagenes(
        carpeta_clase
    )


    print(
        f"Imágenes encontradas: "
        f"{len(imagenes)}"
    )


    if len(imagenes) == 0:

        raise RuntimeError(
            f"No se encontraron imágenes "
            f"en {carpeta_clase}"
        )


    # --------------------------------------------------------
    # Validar dimensiones
    # --------------------------------------------------------

    imagenes_validas = []
    imagenes_invalidas = []


    for imagen in imagenes:

        valida, informacion = (
            validar_imagen(imagen)
        )

        if valida:

            imagenes_validas.append(
                imagen
            )

        else:

            imagenes_invalidas.append(
                (imagen, informacion)
            )


    print(
        f"Imágenes válidas 2048x128: "
        f"{len(imagenes_validas)}"
    )

    print(
        f"Imágenes inválidas: "
        f"{len(imagenes_invalidas)}"
    )


    # --------------------------------------------------------
    # Mostrar algunas imágenes inválidas
    # --------------------------------------------------------

    if imagenes_invalidas:

        print()
        print(
            "Primeras imágenes con "
            "dimensiones incorrectas:"
        )

        for imagen, informacion in (
            imagenes_invalidas[:10]
        ):

            print(
                f"  {imagen} -> "
                f"{informacion}"
            )

        raise RuntimeError(
            f"La clase {clase} contiene "
            "imágenes que no son 2048x128."
        )


    # --------------------------------------------------------
    # Verificar cantidad mínima
    # --------------------------------------------------------

    cantidad_train = (
        CANTIDAD_TRAIN[clase]
    )


    if len(imagenes_validas) < cantidad_train:

        raise RuntimeError(
            f"La clase {clase} tiene "
            f"{len(imagenes_validas)} imágenes, "
            f"pero se necesitan al menos "
            f"{cantidad_train} para train."
        )


    # --------------------------------------------------------
    # Mezclar de forma reproducible
    # --------------------------------------------------------

    imagenes_validas = (
        imagenes_validas.copy()
    )

    rng.shuffle(
        imagenes_validas
    )


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    train = imagenes_validas[
        :cantidad_train
    ]


    # --------------------------------------------------------
    # RESTANTES
    # --------------------------------------------------------

    restantes = imagenes_validas[
        cantidad_train:
    ]


    # --------------------------------------------------------
    # DIVIDIR RESTANTES 50/50
    # --------------------------------------------------------

    punto = len(restantes) // 2


    validation = restantes[
        :punto
    ]


    test = restantes[
        punto:
    ]


    # --------------------------------------------------------
    # DIRECTORIOS
    # --------------------------------------------------------

    carpeta_train = (
        RUTA_SALIDA /
        "train" /
        clase
    )


    carpeta_validation = (
        RUTA_SALIDA /
        "validation" /
        clase
    )


    carpeta_test = (
        RUTA_SALIDA /
        "test" /
        clase
    )


    # --------------------------------------------------------
    # COPIAR
    # --------------------------------------------------------

    copiar_imagenes(
        train,
        carpeta_clase,
        carpeta_train
    )


    copiar_imagenes(
        validation,
        carpeta_clase,
        carpeta_validation
    )


    copiar_imagenes(
        test,
        carpeta_clase,
        carpeta_test
    )


    # --------------------------------------------------------
    # RESUMEN
    # --------------------------------------------------------

    resumen.append(
        {
            "clase": clase,
            "total": len(imagenes_validas),
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
        }
    )


    print(
        f"Train      : {len(train)}"
    )

    print(
        f"Validation : {len(validation)}"
    )

    print(
        f"Test       : {len(test)}"
    )


# ============================================================
# RESUMEN FINAL
# ============================================================

print()
print("=" * 80)
print("RESUMEN FINAL")
print("=" * 80)

total_train = 0
total_validation = 0
total_test = 0


for resultado in resumen:

    print(
        f"{resultado['clase']:20s} "
        f"Train={resultado['train']:4d} | "
        f"Validation={resultado['validation']:4d} | "
        f"Test={resultado['test']:4d} | "
        f"Total={resultado['total']:5d}"
    )


    total_train += (
        resultado["train"]
    )

    total_validation += (
        resultado["validation"]
    )

    total_test += (
        resultado["test"]
    )


print("-" * 80)

print(
    f"{'TOTAL':20s} "
    f"Train={total_train:4d} | "
    f"Validation={total_validation:4d} | "
    f"Test={total_test:4d}"
)


print()
print(
    f"Dataset generado en:\n"
    f"{RUTA_SALIDA}"
)


print()
print("=" * 80)
print("PROCESO TERMINADO")
print("=" * 80)
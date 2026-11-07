# Generación de espectrogramas sintéticos de ecolocalización de murciélagos con DDPM

Código y resultados del proyecto de la Universidad de los Llanos (AdaLab) para generar
espectrogramas sintéticos de pulsos y secuencias de ecolocalización con modelos de difusión
(DDPM, `diffusers`), a partir de grabaciones anotadas en Raven de *Molossus molossus*,
*M. rufus*, *Molossus_1*, *Saccopteryx bilineata*, *S. leptura*, *Saccopteryx* sp. y
*Noctilio albiventris*.

## Estructura

| Carpeta | Contenido |
| --- | --- |
| [`limpieza-dataset/`](limpieza-dataset/) | Pipeline actual de anotaciones y espectrogramas (fases 0–5): inventario, auditoría, normalización y unificación de las anotaciones, y generación de espectrogramas de pulsos y secuencias. Incluye la versión previa de los scripts de limpieza. |
| [`generacion-espectrogramas/`](generacion-espectrogramas/) | Scripts independientes de generación de espectrogramas (versión previa al pipeline) y de selección de muestras de pulsos y secuencias. |
| [`modelos/`](modelos/) | Entrenamiento y evaluación de los modelos: pulsos (DDPM, DiT, barrido de timesteps y learning rate, nSSIM cruzado) y secuencias (por especie y condicionado por especie). |
| [`pseudosecuencias/`](pseudosecuencias/) | Construcción del dataset de pseudo-secuencias de 2 y 3 pulsos con su espaciado real, y entrenamiento multicondicional (especie, número de pulsos, patrón de frecuencia). |
| [`resultados/`](resultados/) | Métricas de los experimentos (CSV, JSON, tabla LaTeX). Sin imágenes. |
| [`datos/`](datos/) | Descripción de los datos del proyecto (anotaciones procesadas, espectrogramas e imágenes generadas), que no se incluyen en el repositorio. |
| [`docs/`](docs/) | `inventario_reorganizacion.csv`: ubicación original y actual de cada archivo del proyecto. |

## Flujo de trabajo

```
Anotaciones Raven (.txt) + audios (.wav)
        │
        ▼
limpieza-dataset/  fase0 → fase1 → fase2 → fase3        anotaciones_unificadas.csv
        │
        ├─ fase4 ──► espectrogramas de pulsos (ventana de 20 ms)
        │              └─► modelos/pulsos/      DDPM de pulsos 128 × 128
        │
        ├─ fase5 ──► espectrogramas de secuencias (ventanas de 1.5 s)
        │              └─► modelos/secuencias/  DDPM por especie y condicionado
        │
        └─► pseudosecuencias/  P0 → P1 → P2 → P3 → P4 → C1   grupos de 2 y 3 pulsos
                       └─► entrenamiento multicondicional (lanzar.py)
```

## Ejecución

Los scripts se ejecutaron en el servidor del laboratorio y tienen rutas absolutas a
`/media/adalab/bodega/CamiloBuitrago/...` (audios, anotaciones y salidas). Para correrlos en otro
equipo hay que cambiar esas rutas:

- `limpieza-dataset/config.py`: `RUTA_BASE`, o la variable de entorno `ANOTACIONES_BASE`.
- `pseudosecuencias/parametros.yaml` y `pseudosecuencias/configs/base_pseudo.yaml`.
- En los scripts de `modelos/` y `generacion-espectrogramas/`, las constantes `RUTA_*` del inicio.

Cada carpeta tiene su propio README con el orden de ejecución y los requisitos.

## Datos

Los audios, las anotaciones y las imágenes no están en este repositorio por su tamaño
(más de 500 MB solo en espectrogramas). [`datos/README.md`](datos/README.md) describe qué hay en
cada carpeta de datos y cómo se generó.

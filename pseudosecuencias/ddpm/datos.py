"""Datasets: carpetas por especie (EspectrogramasDataset) o metadatos con varias condiciones (MetadatosDataset)."""

import random
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def _listar_png(carpeta, recursivo=False):
    patron = carpeta.rglob("*") if recursivo else carpeta.iterdir()
    return sorted(p for p in patron if p.is_file() and p.suffix.lower() == ".png")


class EspectrogramasDataset(Dataset):
    """
    Devuelve {"images": tensor en [-1, 1], "labels": indice de especie}.

    Revisa el tamano de TODAS las imagenes al construirse (solo lee la
    cabecera, es rapido). Si alguna no mide alto x ancho y
    permitir_redimensionar es false, falla con un mensaje claro.
    """

    def __init__(self, cfg_datos, semilla=42):
        self.ruta = Path(cfg_datos["ruta"])
        self.alto = cfg_datos["alto"]
        self.ancho = cfg_datos["ancho"]
        self.canales = cfg_datos.get("canales", 3)
        self.permitir_redimensionar = cfg_datos.get("permitir_redimensionar", False)
        subconjunto = cfg_datos.get("subconjunto")
        especies = cfg_datos.get("especies") or []

        if not self.ruta.is_dir():
            raise FileNotFoundError(f"No existe la carpeta de datos: {self.ruta}")

        self.archivos = []
        if especies:
            self.clases = list(especies)
            for etiqueta, especie in enumerate(especies):
                carpeta = self.ruta / especie
                if not carpeta.is_dir():
                    raise FileNotFoundError(f"No existe la carpeta de la especie '{especie}': {carpeta}")
                pngs = _listar_png(carpeta)
                if not pngs:
                    raise ValueError(f"No hay imagenes .png en {carpeta}")
                pngs = self._submuestrear(pngs, subconjunto, semilla + etiqueta)
                self.archivos += [(p, etiqueta) for p in pngs]
        else:
            self.clases = [self.ruta.name]
            pngs = _listar_png(self.ruta, recursivo=True)
            if not pngs:
                raise ValueError(f"No hay imagenes .png en {self.ruta}")
            self.archivos = [(p, 0) for p in self._submuestrear(pngs, subconjunto, semilla)]

        self.tamanos = Counter()
        for p, _ in self.archivos:
            with Image.open(p) as im:
                self.tamanos[im.size] += 1   # (ancho, alto)

        esperado = (self.ancho, self.alto)
        distintos = {t: n for t, n in self.tamanos.items() if t != esperado}
        if distintos and not self.permitir_redimensionar:
            resumen = ", ".join(f"{h}x{a} px: {n}" for (a, h), n in sorted(distintos.items()))
            raise ValueError(
                f"{sum(distintos.values())} imagenes no miden {self.alto}x{self.ancho} px (alto x ancho): "
                f"{resumen}. Ajustar datos.ancho/datos.alto al tamano real o poner "
                f"datos.permitir_redimensionar: true si el cambio de escala es intencional."
            )

        self.normalizar = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5] * self.canales, [0.5] * self.canales),
        ])

    @staticmethod
    def _submuestrear(rutas, n, semilla):
        if not n or n >= len(rutas):
            return rutas
        return sorted(random.Random(semilla).sample(rutas, n))

    def conteo_por_clase(self):
        conteo = Counter(e for _, e in self.archivos)
        return {self.clases[i]: conteo.get(i, 0) for i in range(len(self.clases))}

    def indices_de_clase(self, etiqueta):
        return [i for i, (_, e) in enumerate(self.archivos) if e == etiqueta]

    def __len__(self):
        return len(self.archivos)

    def __getitem__(self, idx):
        ruta, etiqueta = self.archivos[idx]
        imagen = Image.open(ruta).convert("RGB" if self.canales == 3 else "L")
        if imagen.size != (self.ancho, self.alto):
            imagen = imagen.resize((self.ancho, self.alto), Image.LANCZOS)
        return {"images": self.normalizar(imagen), "labels": torch.tensor(etiqueta, dtype=torch.long)}


def patron_de_transiciones(texto):
    """
    Mismo criterio que p3_grupos.py de pseudosecuencias: 'baja' -> alto-bajo,
    'sube' -> bajo-alto, 'baja;sube' -> alto-bajo-alto, 'sube;baja' -> bajo-alto-bajo,
    todas 'igual' -> constante, cualquier otra combinacion -> irregular.
    Permite usar metadatos renderizados antes de que P3 guardara la columna patron.
    """
    t = [x for x in str(texto).split(";") if x]
    if not t or all(x == "igual" for x in t):
        return "constante"
    if not (all(x in ("sube", "baja") for x in t) and all(a != b for a, b in zip(t, t[1:]))):
        return "irregular"
    nivel = "alto" if t[0] == "baja" else "bajo"
    niveles = [nivel]
    for _ in t:
        nivel = "bajo" if nivel == "alto" else "alto"
        niveles.append(nivel)
    return "-".join(niveles)


class MetadatosDataset(Dataset):
    """
    Dataset multicondicional a partir del metadatos.csv de pseudosecuencias (P4).

    Cada fila es una imagen (ruta_relativa respecto a la carpeta del csv) con sus
    condiciones (p. ej. especie, n, patron). Devuelve {"images", "labels": (k,)}
    donde labels son los indices de cada condicion en su vocabulario.
    """

    def __init__(self, cfg_datos, condiciones, semilla=42):
        import pandas as pd

        self.ruta_meta = Path(cfg_datos["metadatos"])
        if not self.ruta_meta.exists():
            raise FileNotFoundError(f"No existe el archivo de metadatos: {self.ruta_meta}")
        self.base = self.ruta_meta.parent
        self.alto, self.ancho = cfg_datos["alto"], cfg_datos["ancho"]
        self.canales = cfg_datos.get("canales", 3)
        self.permitir_redimensionar = cfg_datos.get("permitir_redimensionar", False)
        self.condiciones = list(condiciones)
        particion = cfg_datos.get("particion", "train")

        df = pd.read_csv(self.ruta_meta)
        self.avisos = []
        if "patron" in self.condiciones and "patron" not in df.columns and "transiciones" in df.columns:
            df["patron"] = df["transiciones"].map(patron_de_transiciones)
            self.avisos.append("metadatos sin columna patron: se calculo a partir de 'transiciones' "
                               "(mismo criterio que p3_grupos.py)")
        faltan = [c for c in self.condiciones + ["ruta_relativa", "especie"] if c not in df.columns]
        if faltan:
            raise ValueError(f"metadatos sin columnas {faltan}; disponibles: {list(df.columns)}")
        if particion and "particion" in df.columns:
            df = df[df["particion"] == particion]
        # especies: las de la configuracion, o todas las que tengan al menos min_imagenes_por_especie
        minimo = cfg_datos.get("min_imagenes_por_especie", 0) or 0
        conteo = df["especie"].value_counts()
        if cfg_datos.get("especies"):
            especies = list(cfg_datos["especies"])
        else:
            especies = sorted(e for e, c in conteo.items() if c >= minimo)
        self.especies_descartadas = {e: int(c) for e, c in conteo.items() if e not in especies}
        if self.especies_descartadas:
            self.avisos.append(f"especies fuera del entrenamiento (menos de {minimo} imagenes en "
                               f"{particion} o no listadas): {self.especies_descartadas}")
        df = df[df["especie"].isin(especies)]
        if df.empty:
            raise ValueError(f"No quedan imagenes en {self.ruta_meta} (particion={particion}, especies={especies})")
        if cfg_datos.get("subconjunto"):
            n = cfg_datos["subconjunto"]
            elegidas = []
            for _, g in df.groupby("especie"):
                elegidas += list(g.sample(min(n, len(g)), random_state=semilla).index)
            df = df.loc[elegidas]
        df = df.sort_values("ruta_relativa").reset_index(drop=True)
        for c in self.condiciones:
            df[c] = df[c].astype(str)

        # vocabulario de cada condicion; la especie respeta el orden de la configuracion
        self.vocab = {c: (list(especies) if c == "especie" else sorted(df[c].unique()))
                      for c in self.condiciones}
        self.vocab = {c: [str(v) for v in vals if str(v) in set(df[c])] for c, vals in self.vocab.items()}
        self.clases = list(self.vocab["especie"]) if "especie" in self.condiciones else sorted(df["especie"].unique())
        self.df = df
        self.etiquetas = torch.tensor(
            [[self.vocab[c].index(v) for c, v in zip(self.condiciones, fila)]
             for fila in df[self.condiciones].itertuples(index=False)], dtype=torch.long)
        # (ruta, indice de especie): misma forma que EspectrogramasDataset, para metricas
        self.archivos = [(self.base / r, self.clases.index(e)) for r, e in zip(df["ruta_relativa"], df["especie"])]

        self.tamanos = Counter()
        for p, _ in self.archivos:
            with Image.open(p) as im:
                self.tamanos[im.size] += 1
        distintos = {t: n for t, n in self.tamanos.items() if t != (self.ancho, self.alto)}
        if distintos and not self.permitir_redimensionar:
            resumen = ", ".join(f"{h}x{a} px: {n}" for (a, h), n in sorted(distintos.items()))
            raise ValueError(f"{sum(distintos.values())} imagenes no miden {self.alto}x{self.ancho} px "
                             f"(alto x ancho): {resumen}")
        self.normalizar = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5] * self.canales, [0.5] * self.canales),
        ])

    def conteo_por_clase(self):
        conteo = Counter(e for _, e in self.archivos)
        return {self.clases[i]: conteo.get(i, 0) for i in range(len(self.clases))}

    def conteo_combinaciones(self):
        return Counter(tuple(f) for f in self.df[self.condiciones].itertuples(index=False))

    def combinaciones_frecuentes(self, k):
        """Las k combinaciones de condiciones mas frecuentes, como tensor (k, n_condiciones)."""
        top = [c for c, _ in self.conteo_combinaciones().most_common(k)]
        return torch.tensor([[self.vocab[c].index(v) for c, v in zip(self.condiciones, comb)] for comb in top],
                            dtype=torch.long)

    def indices_de_clase(self, etiqueta):
        return [i for i, (_, e) in enumerate(self.archivos) if e == etiqueta]

    def etiquetas_de_clase(self, etiqueta):
        return self.etiquetas[self.indices_de_clase(etiqueta)]

    def __len__(self):
        return len(self.archivos)

    def __getitem__(self, idx):
        ruta, _ = self.archivos[idx]
        imagen = Image.open(ruta).convert("RGB" if self.canales == 3 else "L")
        if imagen.size != (self.ancho, self.alto):
            imagen = imagen.resize((self.ancho, self.alto), Image.LANCZOS)
        return {"images": self.normalizar(imagen), "labels": self.etiquetas[idx]}


def crear_dataset(cfg, semilla=42, permitir_redimensionar=None):
    d = dict(cfg["datos"])
    if permitir_redimensionar is not None:
        d["permitir_redimensionar"] = permitir_redimensionar
    if cfg["modo"] == "multicondicional":
        return MetadatosDataset(d, cfg["condiciones"], semilla=semilla)
    return EspectrogramasDataset(d, semilla=semilla)

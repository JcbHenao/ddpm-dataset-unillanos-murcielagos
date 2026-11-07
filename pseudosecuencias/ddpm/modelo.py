"""
UNet2D a partir de la configuracion, y la variante multicondicional.

Multicondicional: cada condicion (especie, n de pulsos, patron de frecuencia, ...)
tiene su propio embedding; los embeddings se SUMAN y entran a la UNet igual que
el embedding de tiempo (class_embed_type="identity"). Al estar separados, una
combinacion poco frecuente (p. ej. una especie con pocos trios alto-bajo-alto)
aprovecha lo aprendido de esa especie con otros patrones y de ese patron en
otras especies.

Cada embedding tiene un indice extra "sin dato". Durante el entrenamiento cada
condicion se reemplaza por "sin dato" con probabilidad prob_sin_condicion: asi
el modelo tambien sabe generar fijando solo algunas condiciones (p. ej. solo la
especie) y se puede usar guia sin clasificador al muestrear.
"""

import json
from pathlib import Path

import torch
from diffusers import UNet2DModel
from torch import nn


def construir_unet(cfg, num_clases=None, embedding_externo=False):
    m, d = cfg["modelo"], cfg["datos"]
    return UNet2DModel(
        sample_size=(d["alto"], d["ancho"]),
        in_channels=d.get("canales", 3),
        out_channels=d.get("canales", 3),
        layers_per_block=m["layers_per_block"],
        block_out_channels=tuple(m["block_out_channels"]),
        down_block_types=tuple(m["down_block_types"]),
        up_block_types=tuple(m["up_block_types"]),
        num_class_embeds=None if embedding_externo else num_clases,
        class_embed_type="identity" if embedding_externo else None,
    )


class Condicionador(nn.Module):
    def __init__(self, tamanos, dim):
        super().__init__()
        # +1: el ultimo indice de cada condicion significa "sin dato"
        self.embeddings = nn.ModuleList(nn.Embedding(t + 1, dim) for t in tamanos)
        for e in self.embeddings:
            nn.init.normal_(e.weight, std=0.02)

    def forward(self, etiquetas):
        return sum(emb(etiquetas[:, i]) for i, emb in enumerate(self.embeddings))


class UNetMulticondicional(nn.Module):
    """Se usa igual que la UNet: modelo(x, t, class_labels=etiquetas (B, k)).sample"""

    def __init__(self, unet, vocab, prob_sin_condicion=0.1):
        super().__init__()
        self.unet = unet
        self.vocab = {k: [str(v) for v in vals] for k, vals in vocab.items()}
        self.nombres = list(self.vocab)
        self.prob_sin_condicion = prob_sin_condicion
        dim = unet.time_embedding.linear_2.out_features
        self.condicionador = Condicionador([len(self.vocab[k]) for k in self.nombres], dim)
        self.register_buffer("indices_nulos",
                             torch.tensor([len(self.vocab[k]) for k in self.nombres], dtype=torch.long),
                             persistent=False)

    @property
    def dtype(self):
        return self.unet.dtype

    def etiquetas_nulas(self, b):
        return self.indices_nulos.unsqueeze(0).expand(b, -1).clone()

    def forward(self, sample, timestep, class_labels=None, return_dict=True):
        etiquetas = class_labels if class_labels is not None else self.etiquetas_nulas(sample.shape[0])
        etiquetas = etiquetas.to(sample.device)
        if self.training and self.prob_sin_condicion > 0:
            quitar = torch.rand(etiquetas.shape, device=etiquetas.device) < self.prob_sin_condicion
            etiquetas = torch.where(quitar, self.indices_nulos.to(etiquetas.device).expand_as(etiquetas), etiquetas)
        emb = self.condicionador(etiquetas)
        return self.unet(sample, timestep, class_labels=emb, return_dict=return_dict)

    def codificar(self, valores, n=1):
        """valores: {condicion: valor o None}. Una condicion ausente o None = 'sin dato'."""
        fila = []
        for k in self.nombres:
            v = valores.get(k)
            if v is None:
                fila.append(len(self.vocab[k]))
            else:
                v = str(v)
                if v not in self.vocab[k]:
                    raise ValueError(f"'{v}' no es un valor conocido de {k}; opciones: {self.vocab[k]}")
                fila.append(self.vocab[k].index(v))
        return torch.tensor([fila] * n, dtype=torch.long)

    def describir(self, fila):
        return ", ".join(f"{k}={self.vocab[k][int(i)] if int(i) < len(self.vocab[k]) else '-'}"
                         for k, i in zip(self.nombres, fila))

    def guardar(self, carpeta):
        carpeta = Path(carpeta)
        self.unet.save_pretrained(carpeta / "unet")
        torch.save(self.condicionador.state_dict(), carpeta / "condicionador.pt")
        with open(carpeta / "condiciones.json", "w", encoding="utf-8") as f:
            json.dump({"vocab": self.vocab, "prob_sin_condicion": self.prob_sin_condicion}, f,
                      indent=2, ensure_ascii=False)

    @classmethod
    def cargar(cls, carpeta, map_location="cpu"):
        carpeta = Path(carpeta)
        with open(carpeta / "condiciones.json", encoding="utf-8") as f:
            info = json.load(f)
        modelo = cls(UNet2DModel.from_pretrained(carpeta / "unet"), info["vocab"], info["prob_sin_condicion"])
        modelo.condicionador.load_state_dict(torch.load(carpeta / "condicionador.pt", map_location=map_location))
        return modelo


def construir_modelo(cfg, dataset):
    """UNet simple, UNet con clase (especie) o UNet multicondicional, segun cfg['modo']."""
    modo = cfg["modo"]
    if modo == "multicondicional":
        unet = construir_unet(cfg, embedding_externo=True)
        return UNetMulticondicional(unet, dataset.vocab,
                                    cfg["entrenamiento"].get("prob_sin_condicion", 0.1))
    return construir_unet(cfg, len(dataset.clases) if modo == "condicional" else None)


def contar_parametros(modelo):
    return sum(p.numel() for p in modelo.parameters())

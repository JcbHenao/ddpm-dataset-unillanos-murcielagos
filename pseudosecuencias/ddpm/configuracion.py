"""
Carga, herencia, validacion y expansion de configuraciones YAML.

Un experimento declara "hereda: ../base.yaml" y solo escribe lo que cambia;
la fusion es recursiva por diccionarios (las listas se reemplazan enteras).
"""

import copy
from pathlib import Path

import yaml

MODOS = {"condicional", "por_especie", "incondicional", "multicondicional"}
ESQUEMAS = {"linear", "squaredcos_cap_v2"}
PREDICCIONES = {"epsilon", "v_prediction"}
PRECISIONES = {"no", "fp16", "bf16"}
MUESTREADORES = {"ddpm", "ddim"}


def _fusionar(base, cambios):
    salida = copy.deepcopy(base)
    for clave, valor in cambios.items():
        if isinstance(valor, dict) and isinstance(salida.get(clave), dict):
            salida[clave] = _fusionar(salida[clave], valor)
        else:
            salida[clave] = copy.deepcopy(valor)
    return salida


def cargar(ruta):
    """Lee un YAML y resuelve su cadena de herencia."""
    ruta = Path(ruta).resolve()
    with open(ruta, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    padre = cfg.pop("hereda", None)
    if padre:
        cfg = _fusionar(cargar((ruta.parent / padre).resolve()), cfg)
    return cfg


def guardar(cfg, ruta):
    with open(ruta, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def validar(cfg):
    """Devuelve la lista de errores (vacia si la configuracion es valida)."""
    errores = []

    def falta(*claves):
        nodo = cfg
        for c in claves:
            if not isinstance(nodo, dict) or c not in nodo:
                errores.append(f"falta la clave {'.'.join(claves)}")
                return True
            nodo = nodo[c]
        return False

    for claves in [("nombre",), ("modo",), ("datos", "alto"), ("datos", "ancho"),
                   ("ruido", "pasos_entrenamiento"), ("entrenamiento", "lote"), ("entrenamiento", "lr")]:
        falta(*claves)
    if errores:
        return errores
    if cfg["modo"] == "multicondicional":
        falta("datos", "metadatos")
        if not cfg.get("condiciones"):
            errores.append("modo 'multicondicional' necesita 'condiciones' (p. ej. [especie, n, patron])")
        p = cfg["entrenamiento"].get("prob_sin_condicion", 0.1)
        if not (0 <= p < 1):
            errores.append("entrenamiento.prob_sin_condicion debe estar en [0, 1)")
        if cfg.get("evaluacion", {}).get("guia", 1.0) > 1 and p == 0:
            errores.append("evaluacion.guia > 1 necesita entrenamiento.prob_sin_condicion > 0 "
                           "(el modelo debe haber visto ejemplos sin condicion)")
    else:
        falta("datos", "ruta")
    if errores:
        return errores

    d, r, e, ev, m = cfg["datos"], cfg["ruido"], cfg["entrenamiento"], cfg.get("evaluacion", {}), cfg["modelo"]
    especies = d.get("especies") or []

    if cfg["modo"] not in MODOS:
        errores.append(f"modo '{cfg['modo']}' no valido; opciones: {sorted(MODOS)}")
    if cfg.get("evaluacion", {}).get("guia", 1.0) > 1 and cfg["modo"] != "multicondicional":
        errores.append("evaluacion.guia > 1 solo esta disponible en modo multicondicional")
    if cfg["modo"] in {"condicional", "por_especie"} and not especies:
        errores.append(f"modo '{cfg['modo']}' necesita datos.especies con al menos una especie")
    if cfg["modo"] == "condicional" and len(especies) < 2:
        errores.append("modo 'condicional' con menos de 2 especies: usar 'incondicional'")
    if d.get("canales") not in (1, 3):
        errores.append("datos.canales debe ser 1 o 3")

    if r.get("esquema") not in ESQUEMAS:
        errores.append(f"ruido.esquema debe ser uno de {sorted(ESQUEMAS)}")
    if r.get("prediccion") not in PREDICCIONES:
        errores.append(f"ruido.prediccion debe ser uno de {sorted(PREDICCIONES)}")
    if r.get("snr_terminal_cero") and r.get("prediccion") != "v_prediction":
        errores.append("ruido.snr_terminal_cero=true exige ruido.prediccion=v_prediction "
                       "(con epsilon el ultimo paso no aporta informacion; Lin et al. 2024)")
    escala = r.get("escala_entrada", 1.0)
    if not (0 < escala <= 1):
        errores.append("ruido.escala_entrada debe estar en (0, 1]")

    if e.get("precision") not in PRECISIONES:
        errores.append(f"entrenamiento.precision debe ser uno de {sorted(PRECISIONES)}")
    if ev.get("muestreador", "ddpm") not in MUESTREADORES:
        errores.append(f"evaluacion.muestreador debe ser uno de {sorted(MUESTREADORES)}")
    pasos_m = ev.get("pasos_muestreo")
    if pasos_m is not None and pasos_m > r["pasos_entrenamiento"]:
        errores.append("evaluacion.pasos_muestreo no puede superar ruido.pasos_entrenamiento")
    if ev.get("muestreador", "ddpm") == "ddpm" and pasos_m not in (None, r["pasos_entrenamiento"]):
        errores.append("con muestreador ddpm, pasos_muestreo debe ser null (todos los pasos); "
                       "para menos pasos usar ddim")

    if any(c % 32 for c in m["block_out_channels"]):
        errores.append("modelo.block_out_channels deben ser multiplos de 32 (GroupNorm de 32 grupos)")
    n_niveles = len(m["block_out_channels"])
    if not (len(m["down_block_types"]) == len(m["up_block_types"]) == n_niveles):
        errores.append("modelo: block_out_channels, down_block_types y up_block_types deben tener el mismo largo")
    divisor = 2 ** (n_niveles - 1)
    if d["alto"] % divisor or d["ancho"] % divisor:
        errores.append(f"datos.alto y datos.ancho deben ser divisibles por {divisor} "
                       f"({n_niveles} niveles en la UNet)")
    return errores


def validar_o_fallar(cfg, origen=""):
    errores = validar(cfg)
    if errores:
        detalle = "\n  - ".join(errores)
        raise ValueError(f"Configuracion invalida {origen}:\n  - {detalle}")


def expandir(cfg):
    """
    Convierte una configuracion en la lista de corridas que genera.
    modo por_especie -> una corrida incondicional por especie, cada una con su carpeta.
    """
    if cfg["modo"] != "por_especie":
        return [copy.deepcopy(cfg)]
    corridas = []
    for especie in cfg["datos"]["especies"]:
        c = copy.deepcopy(cfg)
        c["modo"] = "incondicional"
        c["datos"]["especies"] = [especie]
        c["nombre"] = f"{cfg['nombre']}-{especie}"
        c["experimento_padre"] = cfg["nombre"]
        corridas.append(c)
    return corridas


def lote_efectivo(cfg, n_gpus=1):
    e = cfg["entrenamiento"]
    return e["lote"] * e.get("acumulacion", 1) * n_gpus

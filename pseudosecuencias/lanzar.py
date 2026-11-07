"""
Ejecuta una cola de experimentos, uno tras otro, cada uno en su propia
carpeta runs/<nombre>/ y en su propio proceso (la GPU queda libre al terminar).

Ejemplos
    python lanzar.py configs/experimentos/02_secuencias_cond_lineal1500.yaml
    python lanzar.py configs/experimentos/                 # todos los .yaml de la carpeta, en orden
    python lanzar.py configs/experimentos/04_*.yaml --gpus 2
    python lanzar.py configs/experimentos/ --dispositivos 1   # solo la GPU 1
    python lanzar.py configs/experimentos/ --listar           # muestra la cola sin ejecutar

Reglas (el nombre del experimento identifica la corrida)
    - Completada con la misma configuracion: se salta (--repetir la vuelve a correr).
    - A medias con la misma configuracion: se reanuda desde su ultimo checkpoint.
    - Ya existe con OTRA configuracion: la cola no arranca; cambiar el nombre en el YAML
      o usar --desde-cero.
    - --desde-cero y --repetir mueven la carpeta anterior a runs/_archivo/ (no se borra nada).
    - Si una corrida falla, se registra el error y la cola sigue con la siguiente.
    - Una corrida que otro lanzar.py esta entrenando en ese momento se salta (archivo .bloqueo).
"""

import argparse
import datetime as dt
import os
import shutil
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))

from ddpm import configuracion, registro  # noqa: E402


def recolectar_configs(rutas):
    archivos = []
    for r in rutas:
        p = Path(r)
        if p.is_dir():
            archivos += sorted(p.glob("*.yaml"))
        elif p.is_file():
            archivos.append(p)
        else:
            sys.exit(f"No existe: {r}")
    return archivos


def armar_cola(archivos):
    cola = []
    for archivo in archivos:
        cfg = configuracion.cargar(archivo)
        configuracion.validar_o_fallar(cfg, f"({archivo})")
        for corrida in configuracion.expandir(cfg):
            corrida["archivo_origen"] = str(Path(archivo).resolve())
            cola.append(corrida)
    nombres = [c["nombre"] for c in cola]
    repetidos = {n for n in nombres if nombres.count(n) > 1}
    if repetidos:
        sys.exit(f"Hay nombres de corrida repetidos en la cola: {sorted(repetidos)}")
    return cola


def comando(run_dir, gpus):
    if gpus > 1:
        return [sys.executable, "-m", "accelerate.commands.launch", "--multi_gpu",
                "--num_processes", str(gpus), "--num_machines", "1", "--dynamo_backend", "no",
                "-m", "ddpm.entrenar", "--run-dir", str(run_dir)]
    return [sys.executable, "-m", "ddpm.entrenar", "--run-dir", str(run_dir)]


def config_cambio(cfg, run_dir):
    """True si ya existe una corrida con ese nombre pero con otra configuracion."""
    ruta = run_dir / "config.yaml"
    if not ruta.exists():
        return False
    anterior = configuracion.cargar(ruta)
    anterior.pop("archivo_origen", None)
    return anterior != {k: v for k, v in cfg.items() if k != "archivo_origen"}


def decidir(cfg, carpeta_runs, args):
    """
    completada y misma config  -> saltar        (--repetir: archivar y repetir)
    config distinta            -> conflicto     (--desde-cero o --repetir: archivar y correr)
    incompleta y misma config  -> reanudar      (--desde-cero: archivar y correr limpia)
    no existe                  -> nueva
    """
    run_dir = carpeta_runs / cfg["nombre"]
    if not run_dir.exists():
        return "nueva"
    if config_cambio(cfg, run_dir):
        return "archivar" if (args.desde_cero or args.repetir) else "conflicto"
    if registro.leer_estado(run_dir).get("estado") == "completado":
        return "archivar" if args.repetir else "saltar"
    return "archivar" if args.desde_cero else "reanudar"


def bloqueada(run_dir):
    """PID de otro lanzar.py que esta entrenando esta misma corrida, o None."""
    try:
        pid = int((run_dir / ".bloqueo").read_text().strip())
    except (FileNotFoundError, ValueError):
        return None
    try:
        os.kill(pid, 0)          # no envia nada: solo comprueba si el proceso existe
        return pid if pid != os.getpid() else None
    except OSError:
        return None              # bloqueo huerfano de una corrida que murio


def ejecutar(cfg, carpeta_runs, args):
    run_dir = carpeta_runs / cfg["nombre"]
    pid = bloqueada(run_dir)
    if pid:
        print(f"[ocupada]  {cfg['nombre']}: la esta entrenando otro proceso (PID {pid}); se salta")
        return "ocupada"
    accion = decidir(cfg, carpeta_runs, args)

    if accion == "saltar":
        print(f"[saltada]  {cfg['nombre']} (ya completada)")
        return "saltada"

    if accion == "archivar":
        destino = carpeta_runs / "_archivo" / f"{cfg['nombre']}__{dt.datetime.now():%Y%m%d_%H%M%S}"
        destino.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(run_dir), str(destino))
        print(f"[archivo]  corrida anterior movida a {destino}")

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / ".bloqueo").write_text(str(os.getpid()))
    ruta_config = run_dir / "config.yaml"
    configuracion.guardar(cfg, ruta_config)
    registro.escribir_json(run_dir / "entorno.json", registro.info_entorno())
    registro.actualizar_estado(run_dir, estado="en_cola", lanzado=registro.ahora(), gpus=args.gpus)

    entorno = dict(os.environ)
    entorno["PYTHONPATH"] = str(RAIZ) + os.pathsep + entorno.get("PYTHONPATH", "")
    entorno.setdefault("TQDM_MININTERVAL", "30")   # barras de progreso menos ruidosas en salida.log
    if args.dispositivos is not None:
        entorno["CUDA_VISIBLE_DEVICES"] = args.dispositivos

    print(f"\n[inicio]   {cfg['nombre']}  ->  {run_dir}")
    with open(run_dir / "salida.log", "a", encoding="utf-8") as log:
        log.write(f"\n===== {registro.ahora()} =====\n")
        proceso = subprocess.Popen(comando(run_dir, args.gpus), cwd=RAIZ, env=entorno,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for linea in proceso.stdout:
                sys.stdout.write(linea)
                log.write(linea)
            codigo = proceso.wait()
        except KeyboardInterrupt:
            proceso.terminate()
            proceso.wait()
            (run_dir / ".bloqueo").unlink(missing_ok=True)
            registro.actualizar_estado(run_dir, estado="interrumpido", fin=registro.ahora())
            raise

    (run_dir / ".bloqueo").unlink(missing_ok=True)
    final = registro.leer_estado(run_dir).get("estado")
    if codigo == -9 and final != "completado":
        registro.actualizar_estado(run_dir, estado="fallido", fin=registro.ahora(),
                                   error="el sistema mato el proceso (codigo -9), casi siempre por falta de "
                                         "memoria: bajar entrenamiento.lote y subir acumulacion")
        final = "fallido"
    elif codigo != 0 and final != "fallido":
        registro.actualizar_estado(run_dir, estado="fallido", fin=registro.ahora(),
                                   error=f"el proceso termino con codigo {codigo}; ver salida.log")
        final = "fallido"
    print(f"[{final}]  {cfg['nombre']}")
    return final


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("configs", nargs="+", help="archivos .yaml o carpetas con .yaml")
    parser.add_argument("--carpeta-runs", default=str(RAIZ / "runs"))
    parser.add_argument("--gpus", type=int, default=1, help="GPUs por corrida (>1 usa accelerate --multi_gpu)")
    parser.add_argument("--dispositivos", default=None, help="CUDA_VISIBLE_DEVICES para la cola, p. ej. '0' o '0,1'")
    parser.add_argument("--desde-cero", action="store_true", help="archivar corridas incompletas y empezar limpio")
    parser.add_argument("--repetir", action="store_true", help="volver a correr tambien las completadas")
    parser.add_argument("--listar", action="store_true", help="mostrar la cola y salir")
    args = parser.parse_args()

    cola = armar_cola(recolectar_configs(args.configs))
    carpeta_runs = Path(args.carpeta_runs).resolve()

    print(f"Cola: {len(cola)} corrida(s)  |  runs en {carpeta_runs}")
    conflictos = []
    for c in cola:
        estado = registro.leer_estado(carpeta_runs / c["nombre"]).get("estado", "nueva")
        accion = decidir(c, carpeta_runs, args)
        if accion == "conflicto":
            conflictos.append(c["nombre"])
        r = c["ruido"]
        print(f"  - {c['nombre']:<50} [{estado} -> {accion}]  modo={c['modo']} pasos={r['pasos_entrenamiento']} "
              f"esquema={r['esquema']} lr={c['entrenamiento']['lr']} "
              f"lote_ef={configuracion.lote_efectivo(c, args.gpus)}")
    if conflictos:
        sys.exit(
            "\nEstas corridas ya existen con OTRA configuracion: " + ", ".join(conflictos) +
            "\nPara no mezclar resultados: cambiar 'nombre' en el YAML (recomendado), "
            "o usar --desde-cero para archivar la anterior y empezar de nuevo."
        )
    if args.listar:
        return

    resultados = {}
    try:
        for cfg in cola:
            resultados[cfg["nombre"]] = ejecutar(cfg, carpeta_runs, args)
    except KeyboardInterrupt:
        print("\nCola interrumpida por el usuario.")
    finally:
        from resumir import resumir
        ruta = resumir(carpeta_runs)
        print("\nResultado de la cola:")
        for nombre, estado in resultados.items():
            print(f"  {estado:<12} {nombre}")
        if ruta:
            print(f"Resumen de todas las corridas: {ruta}")


if __name__ == "__main__":
    main()

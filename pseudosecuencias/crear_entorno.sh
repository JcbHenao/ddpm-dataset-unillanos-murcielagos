#!/usr/bin/env bash
# Crea el entorno virtual del proyecto de pseudo-secuencias (.venv-pseudo),
# separado del entorno del pipeline de limpieza y del de entrenamiento.
#
#   bash crear_entorno.sh                     # usa python3
#   PYTHON=python3.11 bash crear_entorno.sh   # otra version de Python
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"

if ! command -v "$PY" >/dev/null 2>&1; then
  echo "No se encontro $PY. Indicar otro con: PYTHON=/ruta/a/python3 bash crear_entorno.sh"
  exit 1
fi
if ! "$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)'; then
  echo "Se necesita Python 3.9 o mas nuevo; $PY es $("$PY" --version 2>&1)."
  exit 1
fi
if ! "$PY" -c 'import venv, ensurepip' >/dev/null 2>&1; then
  echo "A $PY le falta el modulo venv/ensurepip. En Ubuntu: sudo apt install python3-venv"
  echo "(o crear el entorno con conda; ver README, seccion Instalacion)."
  exit 1
fi
if [ -d .venv-pseudo ]; then
  echo "Ya existe .venv-pseudo. Para recrearlo: rm -rf .venv-pseudo && bash crear_entorno.sh"
  exit 1
fi

"$PY" -m venv .venv-pseudo
# shellcheck disable=SC1091
source .venv-pseudo/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
# versiones exactas instaladas, para poder reproducir el entorno en otra maquina
pip freeze > requirements.lock.txt

echo
echo "Entorno listo: $(python --version) en $(pwd)/.venv-pseudo"
echo "Activar con:   source $(pwd)/.venv-pseudo/bin/activate"
echo "Siguiente:     python p0_verificar.py"

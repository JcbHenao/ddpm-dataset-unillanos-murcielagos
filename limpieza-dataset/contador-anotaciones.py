import os
from collections import defaultdict

ruta = "/media/adalab/bodega/CamiloBuitrago/anotaciones"
extension = ".txt"

archivos = defaultdict(list)

for raiz, _, nombres in os.walk(ruta):
    for archivo in nombres:
        if archivo.lower().endswith(extension):
            base = os.path.splitext(archivo)[0]
            archivos[base].append(os.path.join(raiz, archivo))

total = sum(len(v) for v in archivos.values())
unicos = len(archivos)

print(f"Total archivos: {total}")
print(f"Nombres únicos: {unicos}")

duplicados = {k: v for k, v in archivos.items() if len(v) > 1}

print(f"Duplicados: {len(duplicados)}")

for nombre, rutas in list(duplicados.items())[:10]:
    print(f"\n{nombre} ({len(rutas)} veces)")
    for r in rutas:
        print(f"  - {r}")
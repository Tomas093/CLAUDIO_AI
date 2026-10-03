"""Arma un dataset con build_all.py y una receta (30/09). Mismos tamanos que ds21 (hook N21 de train.py).
    py -3 -u src/armar_ds.py ds22 receta_ds22
"""
import os, sys, subprocess, importlib
sys.path.insert(0, os.path.dirname(__file__))
from paths import WORK
nombre, receta = sys.argv[1], sys.argv[2]
RECETA = importlib.import_module(receta).RECETA
env = dict(os.environ, NP=os.environ.get('L_NP', '12000'), NN=os.environ.get('L_NN', '3500'),
           NR=os.environ.get('L_NR', '900'), USAR_PSEUDO='1', PSEUDO_ALTA=os.environ.get('PSEUDO_ALTA', '0.80'),
           CLAUDIO_DS=os.path.join(WORK, nombre))
env.update({k: os.environ.get(k, v) for k, v in RECETA.items()})
subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), 'build_all.py')], env=env, check=True)

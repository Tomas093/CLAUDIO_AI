import os, sys, subprocess
sys.path.insert(0, 'src')
from receta_ds21 import RECETA
env = dict(os.environ, NP='800', NN='200', NR='160', USAR_PSEUDO='1', PSEUDO_ALTA='0.80',
           CLAUDIO_DS=os.path.abspath('work/ds21_muestra'))
env.update(RECETA)
sys.exit(subprocess.run([sys.executable, 'src/build_all.py'], env=env).returncode)

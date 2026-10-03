"""ds26 (30/09): ds23 (sintetico + clase auxiliar de negativos, tiles manuales/reales viejos) + TILES REALES NUEVOS
del GT de Tomas (work/reales_tomas, planos que no son de test) repetidos x REP_RT en train. Val: la de ds23 + la de
reales_tomas (planos apartados). Hardlinks.   py -3 src/armar_ds26.py"""
import os, sys, glob, shutil
sys.path.insert(0, os.path.dirname(__file__))
from paths import WORK
A, R, D = os.path.join(WORK, 'ds23'), os.path.join(WORK, 'reales_tomas'), os.path.join(WORK, os.environ.get('DS_OUT', 'ds26')); K = int(os.environ.get('REP_RT', '2'))
def ln(a, b):
    try: os.link(a, b)
    except OSError: shutil.copy2(a, b)
n = {'train': 0, 'val': 0}
for sp in ('train', 'val'):
    for t in ('images', 'labels'): os.makedirs(os.path.join(D, t, sp), exist_ok=True)
    for src, reps in ((A, 1), (R, K if sp == 'train' else 1)):
        for f in glob.glob(os.path.join(src, 'images', sp, '*.png')):
            nom = os.path.basename(f); lab = os.path.join(src, 'labels', sp, nom[:-4] + '.txt')
            for k in range(reps):
                q = nom if k == 0 else '%s_r%d.png' % (nom[:-4], k)
                ln(f, os.path.join(D, 'images', sp, q))
                if os.path.exists(lab): ln(lab, os.path.join(D, 'labels', sp, q[:-4] + '.txt'))
                n[sp] += 1
open(os.path.join(D, 'data.yaml'), 'w').write(open(os.path.join(A, 'data.yaml')).read().replace('ds23', os.path.basename(D)))
print('[ds26]', n)

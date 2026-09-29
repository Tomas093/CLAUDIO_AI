"""Orquestador autonomo: biblioteca completa -> dataset -> C (prueba corta) -> experimentos largos -> evaluacion -> elige el mejor.
Cada paso se registra en work/reporte/estado.json. Si un experimento falla, sigue con el siguiente."""
import sys, os, json, subprocess, time, shutil, traceback
sys.path.insert(0, os.path.dirname(__file__))
from paths import WORK, PACK, DS
PY = sys.executable; SRC = os.path.dirname(__file__)
REP = os.path.join(WORK, 'reporte'); os.makedirs(REP, exist_ok=True)
EST = os.path.join(REP, 'estado.json')
estado = json.load(open(EST)) if os.path.exists(EST) else {}
def log(k, v):
    estado[k] = v; estado['ultima_actualizacion'] = time.strftime('%Y-%m-%d %H:%M:%S')
    json.dump(estado, open(EST, 'w'), indent=1, ensure_ascii=False); print('[noche]', k, v, flush=True)
def run(key, args, env=None):
    if estado.get(key, {}).get('ok'): print('[noche] ya hecho', key); return True
    t = time.time(); e = dict(os.environ, **(env or {}))
    r = subprocess.run([PY, *args], env=e)
    log(key, dict(ok=r.returncode == 0, minutos=round((time.time()-t)/60, 1)))
    return r.returncode == 0

run('1_biblioteca_completa', [os.path.join(SRC, 'build_lib_full.py')])
if not os.path.exists(os.path.join(DS, 'data.yaml')):
    run('2_dataset', [os.path.join(SRC, 'build_all.py')], env=dict(NP='22000', NN='3000', NR='800'))
EXPS = [  # nombre, modelo, epocas, horas maximas
    ('C_rapido_n20', 'yolo11n.pt', 20, 0),
    ('A_n_largo', 'yolo11n.pt', 150, 4.0),
    ('B_s_largo', 'yolo11s.pt', 120, 4.5),
]
res = {}
for name, model, ep, hrs in EXPS:
    ok = run(f'3_train_{name}', [os.path.join(SRC, 'train.py'), '--model', model, '--name', name, '--epochs', str(ep), '--hours', str(hrs)])
    w = os.path.join(WORK, 'runs', name, 'best_limpio.pt')
    if ok and os.path.exists(w):
        run(f'4_eval_{name}', [os.path.join(SRC, 'evaluate.py'), w, '0.20', name])
        rj = os.path.join(WORK, 'eval', name, 'resumen.json')
        if os.path.exists(rj):
            R = json.load(open(rj))
            tot_gt = sum(v['gt'] for v in R.values()); tot_tp = sum(v['tp'] for v in R.values()); tot_fp = sum(v['fp'] for v in R.values())
            res[name] = dict(recall_total=round(tot_tp/tot_gt, 4), fp_total=tot_fp, por_plano={k: (v['recall'], v['precision'], v['umbral_max_recall100']) for k, v in R.items()})
            log('resultados', res)
if res:
    best = max(res, key=lambda k: (res[k]['recall_total'], -res[k]['fp_total']))
    shutil.copy(os.path.join(WORK, 'runs', best, 'best_limpio.pt'), os.path.join(PACK, 'best_componente_v2.pt'))
    log('mejor', dict(experimento=best, **res[best]))
log('FIN', True)

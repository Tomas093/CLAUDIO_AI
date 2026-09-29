"""Aplica el verificador (src/verif_train.py) a las detecciones ya guardadas de una evaluacion (29/09).

Para cada plano: renderiza el DXF igual que evaluate.py (escala automatica, darken) a la escala VERIF_ESCALA
(1,6 por defecto: el texto queda en ~18 px, dentro del rango de los tiles de entrenamiento), recorta cada
deteccion con el mismo contexto que en verif_datos.recorte y le agrega la probabilidad `p` del verificador.
Escribe work/eval/<tag_salida>_{005,marcelo}/<plano>_detecciones.csv con columnas x1,y1,x2,y2,conf,p y conf
reemplazada por la puntuacion combinada (VERIF_COMB: 'p' | 'conf*p' | 'raiz' = sqrt(conf*p)).

    venv_rfdetr/Scripts/python.exe -u src/verif_aplicar.py work/verif/verif_v1.pt V5_RF4_nano V6_RF4v1
"""
import os, sys, csv, importlib
import cv2, numpy as np, torch, ezdxf
sys.path.insert(0, os.path.dirname(__file__))
from paths import BASE, WORK
from render import render_doc, cad2px
from scale import auto_ppc
from postproc import darken
from verif_datos import recorte
from verif_train import Verif, entrada


def main():
    pesos, tag_in, tag_out = sys.argv[1], sys.argv[2], sys.argv[3]
    esc = float(os.environ.get('VERIF_ESCALA', '1.6')); comb = os.environ.get('VERIF_COMB', 'p')
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = Verif().to(dev); net.load_state_dict(torch.load(pesos, map_location=dev)); net.eval()
    import evaluate as ev
    for suf, qet in (('005', False), ('marcelo', True)):
        if qet: os.environ['EVAL_SET'] = 'qet'; os.environ['QET_DIR'] = 'test_marcelo'
        else: os.environ.pop('EVAL_SET', None)
        importlib.reload(ev)
        od = os.path.join(WORK, 'eval', '%s_%s' % (tag_out, suf)); os.makedirs(od, exist_ok=True)
        for pl, dxf, gtp in ev.PLANOS:
            f = os.path.join(WORK, 'eval', '%s_%s' % (tag_in, suf), pl + '_detecciones.csv')
            dets = [[float(r[k]) for k in ('x1', 'y1', 'x2', 'y2', 'conf')] for r in csv.DictReader(open(f))]
            doc = ezdxf.readfile(os.path.join(BASE, dxf)); ppc = auto_ppc(doc)
            img, meta = render_doc(doc, ppc * esc); img = darken(img)
            X, M = [], []
            for d in dets:
                a = cad2px(meta, d[0], d[3]); b = cad2px(meta, d[2], d[1])
                c, m = recorte(img, [a[0], a[1], b[0], b[1]]); X.append(c); M.append(m)
            p = []
            with torch.no_grad():
                for i in range(0, len(X), 1024):
                    xb = torch.tensor(np.array(X[i:i + 1024]), device=dev); mb = torch.tensor(np.array(M[i:i + 1024], np.float32), device=dev)
                    p.append(torch.sigmoid(net(entrada(xb, mb))).cpu().numpy())
            p = np.concatenate(p) if p else np.zeros(0)
            with open(os.path.join(od, pl + '_detecciones.csv'), 'w', newline='') as h:
                w = csv.writer(h); w.writerow(['x1', 'y1', 'x2', 'y2', 'conf', 'p', 'conf_det'])
                for d, pi in zip(dets, p):
                    s = pi if comb == 'p' else d[4] * pi if comb == 'conf*p' else float(np.sqrt(d[4] * pi))
                    w.writerow([*d[:4], round(float(s), 4), round(float(pi), 4), d[4]])
            print('[verif] %s: %d detecciones' % (pl, len(dets)), flush=True)


if __name__ == '__main__':
    main()

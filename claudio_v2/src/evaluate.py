"""Evalua pesos sobre los DXF de test completos (con texto). Escala automatica por altura de texto.
Salida: work/eval/<plano>_visual.png (verde=TP, rojo=FN, azul=FP), CSV de detecciones y resumen.json"""
import sys, os, json, csv, cv2, numpy as np, ezdxf
sys.path.insert(0, os.path.dirname(__file__))
from paths import BASE, WORK, PACK
from render import render_doc, px2cad, cad2px
from scale import auto_ppc
from postproc import darken
try:
    from ultralytics import YOLO
except ImportError:          # entorno de RF-DETR 1.11 (venv311 en D:): sin ultralytics
    YOLO = None

PLANOS = [
    ('test1', 'test1.dxf', 'test/test_1/verdad_terreno/test1_completo.csv'),
    ('test_2', 'test_2.dxf', 'test/test_2/verdad_terreno/test_2_completo.csv'),
    ('fl_un_02', 'dxf/FL-UN-02_tablero_1.dxf', 'dxf/fl_un_02_gt_completo.csv'),
    ('tsss_2', 'TSSS_2 (1).dxf', 'dxf/tsss_2_gt_completo.csv'),
]
if os.environ.get('EVAL_SET') == 'qet':   # test extra: folios QElectroTech (solo test, no entrenamiento)
    import glob as _g
    import zipfile as _z
    _root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    if not os.path.isdir(os.path.join(_root, 'test_qet')) and os.path.exists(os.path.join(_root, 'test_qet.zip')):
        _z.ZipFile(os.path.join(_root, 'test_qet.zip')).extractall(os.path.join(_root, 'test_qet'))
    _d = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', os.environ.get('QET_DIR','test_qet'))
    PLANOS = [(os.path.basename(p)[:-4], os.path.relpath(p, BASE), os.path.relpath(p[:-4] + '_gt.csv', BASE)) for p in sorted(_g.glob(os.path.join(_d, '*.dxf')))]
# 5 Cajas en T de test1 que faltan en el CSV (centro del recuadro 'T')
EXTRA = {'test1': [(x + .12, 1851.223 + .18) for x in (1890.217, 1901.183, 1912.149, 1923.116, 1934.082)]}

def load_gt(name, path):
    # 20/09: con GT_V2=1 se usa el csv corregido por la auditoria (`<nombre>_v2.csv`), que
    # agrega los 221 componentes que al GT le faltaban. Sobre LU-UN-01 y nyw-un-01 le faltaba
    # el 17%: eso no solo infla el recall (lo que el GT no tiene no se le reclama a nadie),
    # tambien hunde la precision, porque cuando un modelo SI detecta uno de esos hoy se le
    # cuenta como falso positivo. Se deja opcional para poder medir con los dos y comparar.
    # Cascada: _v3 (completo + cenido) > _v2 (completo) > original. Con GT_V3=1 se toma la
    # mejor version disponible de cada plano; los que no tienen _v3 (porque no tenian el
    # problema de las cajas infladas) caen solos al _v2.
    # 20/09: GT_V4=1 agrega la correccion de `snap_audit_gt.py`. Las filas que agrego la
    # auditoria traian una caja del tamano tipico del plano (el revisor solo dio el centro) y
    # venian duplicadas: los tiles de la revision se solapan, asi que un componente del borde
    # se marco dos veces. Como el matching es 1-a-1 por distancia de centros, cada duplicado
    # era un FN que ningun modelo podia evitar. _v4 ajusta la caja al dibujo real y fusiona.
    # 25/09: GT_V5=1 = v4 + faltantes que marco Tomas (test1: los 5 fusibles "x3" de las lamparas,
    # dibujados con lineas sueltas, sin bloque). Cascada v5 > v4 > v3 > v2.
    # 28/09: GT_V6=1 = v5 sin los 9 `AUDIT-rotulo_con_nombre` de test1. Tomas: los rectangulos altos
    # con el nombre del tablero en vertical ("TSASC4-6 LUZ CAB.") NO son componentes; los habia
    # agregado la auditoria automatica del 20/09. Cascada v6 > v5 > v4 > v3 > v2.
    # 29/09: GT_V7=1 = v6 + 289 componentes que faltaban (auditoria de los FP con conf >= 0,3 de RF4 con
    # subagentes y las reglas de Tomas; casi todos en LU-UN-01 y nyw-un-01: cajas BA, interruptores, TOMA...).
    # PROVISORIO hasta que Tomas apruebe la grilla (work/auditoria_gt_v7).
    # 30/09: GT_V8=1 = v7 de la otra sesion (sin PAT, testigos divididos, EZE/test1 reajustados) + mis AUDIT7 con la
    # revision de Tomas (TOMA = gabinete entero, BA y accesorios si, reservas recortadas, KC completo, cajas
    # corridas realineadas). Ver src/gt_v8.py. Las zonas neutras (<gt>_v8_neutras.csv) las filtra neutras().
    # 30/09: GT_V9=1 = GT nuevo de Tomas (Ground-Truth-Planos.zip, reemplaza a todos los anteriores) + la tabla
    # REFERENCIAS como zona neutra (sus criterios: "cuadros de REFERENCIAS: fuera"). Neutras en <gt>_v9_neutras.csv.
    # 03/10: GT_V10=1 = GT actualizado de Tomas del 03/10 (Ground-Truth-Planos.zip, LU 1283 / nyw 1084 / contactores test_2)
    # + la tabla REFERENCIAS neutra (de v9). Neutras en <gt>_v10_neutras.csv.
    if os.environ.get('GT_V10') == '1':
        cand = path[:-4] + '_v10.csv'
        if os.path.exists(cand): path = cand
    elif os.environ.get('GT_V9') == '1':
        cand = path[:-4] + '_v9.csv'
        if os.path.exists(cand): path = cand
    elif os.environ.get('GT_V8') == '1':
        for suf in ('_v8.csv', '_v7.csv', '_v6.csv', '_v5.csv', '_v4.csv', '_v3.csv', '_v2.csv'):
            cand = path[:-4] + suf
            if os.path.exists(cand):
                path = cand
                break
    elif os.environ.get('GT_V7') == '1':
        for suf in ('_v7.csv', '_v6.csv', '_v5.csv', '_v4.csv', '_v3.csv', '_v2.csv'):
            cand = path[:-4] + suf
            if os.path.exists(cand):
                path = cand
                break
    elif os.environ.get('GT_V6') == '1':
        for suf in ('_v6.csv', '_v5.csv', '_v4.csv', '_v3.csv', '_v2.csv'):
            cand = path[:-4] + suf
            if os.path.exists(cand):
                path = cand
                break
    elif os.environ.get('GT_V5') == '1':
        for suf in ('_v5.csv', '_v4.csv', '_v3.csv', '_v2.csv'):
            cand = path[:-4] + suf
            if os.path.exists(cand):
                path = cand
                break
    elif os.environ.get('GT_V4') == '1':
        for suf in ('_v4.csv', '_v3.csv', '_v2.csv'):
            cand = path[:-4] + suf
            if os.path.exists(cand):
                path = cand
                break
    elif os.environ.get('GT_V3') == '1':
        for suf in ('_v3.csv', '_v2.csv'):
            cand = path[:-4] + suf
            if os.path.exists(cand):
                path = cand
                break
    elif os.environ.get('GT_V2') == '1':
        v2 = path[:-4] + '_v2.csv'
        if os.path.exists(v2):
            path = v2
    gt = []
    with open(path, newline='', encoding='utf-8', errors='ignore') as f:
        for r in csv.DictReader(f):
            if r.get('x1'):
                gt.append(dict(c=(float(r['x_cad']), float(r['y_cad'])), b=(float(r['x1']), float(r['y1']), float(r['x2']), float(r['y2'])), n=r['block_name']))
            else:
                gt.append(dict(c=(float(r['x_cad']), float(r['y_cad'])), b=None, n=r['block_name']))
    if not any(g['n'].startswith('Caja-T') for g in gt):
      for c in EXTRA.get(name, []): gt.append(dict(c=c, b=(c[0]-.3, c[1]-.3, c[0]+.3, c[1]+.3), n='Caja en T'))
    return gt

def neutras(path):
    """Zonas neutras del GT v8 (ni TP ni FP ni FN): cajas [(x1,y1,x2,y2)]. Vacio si no hay GT_V8."""
    ver = 'v10' if os.environ.get('GT_V10') == '1' else 'v9' if os.environ.get('GT_V9') == '1' else 'v8' if os.environ.get('GT_V8') == '1' else None
    p = path[:-4] + '_%s_neutras.csv' % ver
    if ver is None or not os.path.exists(p): return []
    return [tuple(float(r[k]) for k in ('x1', 'y1', 'x2', 'y2')) for r in csv.DictReader(open(p, encoding='utf-8'))]


def fuera_de_neutras(dets, zonas):
    """Saca las detecciones cuyo centro cae en una zona neutra (margen 20%)."""
    if not zonas: return dets
    def en(d, z):
        # margen 20% pero con tope: en zonas grandes (tabla REFERENCIAS) el 20% se comia componentes vecinos
        mx, my = min((z[2] - z[0]) * .2, .1), min((z[3] - z[1]) * .2, .1); cx, cy = (d[0] + d[2]) / 2, (d[1] + d[3]) / 2
        return z[0] - mx <= cx <= z[2] + mx and z[1] - my <= cy <= z[3] + my
    return [d for d in dets if not any(en(d, z) for z in zonas)]

class _Cajas:
    def __init__(self, xyxy, conf, cls=None): self.xyxy, self.conf, self.cls = xyxy, conf, cls


class _Res:
    def __init__(self, det):
        import torch
        self.boxes = _Cajas(torch.as_tensor(det.xyxy, dtype=torch.float32), torch.as_tensor(det.confidence, dtype=torch.float32),
                            torch.as_tensor(det.class_id if det.class_id is not None else [0] * len(det.xyxy)))


class RFDETRComoYOLO:
    """24/09. RF-DETR (pesos .pth, corre en venv_rfdetr) con la interfaz de `YOLO.predict` que usa
    `infer`, asi se evalua con exactamente la misma receta (tiles, 2 escalas, fusion en CAD)."""
    def __init__(self, w):
        import rfdetr as _rf
        RFDETRNano = {'nano': _rf.RFDETRNano, 'small': _rf.RFDETRSmall, 'medium': _rf.RFDETRMedium}[os.environ.get('RFDETR_VAR', 'nano')]   # 01/10
        # 28/09: ruta absoluta; rfdetr 1.11 busca las relativas en ~/.roboflow/models (la eval de RF5 fallo por eso)
        self.m = RFDETRNano(pretrain_weights=os.path.abspath(w), resolution=int(os.environ.get('RFDETR_RES', '640')))
        try: self.m.optimize_for_inference()
        except Exception as e: print('[rfdetr] sin optimize_for_inference:', e)

    def predict(self, batch, imgsz=640, conf=.05, verbose=False):
        det = self.m.predict([b[:, :, ::-1].copy() for b in batch], threshold=conf)   # BGR -> RGB
        return [_Res(d) for d in (det if isinstance(det, list) else [det])]


FUERA = {int(k) for k in os.environ.get('EVAL_CLASES_FUERA', '').split(',') if k}

def infer(model, img, conf, tile=640, stride=320):
    H, W = img.shape; dets = []
    ys = list(range(0, max(1, H - tile) + 1, stride)); xs = list(range(0, max(1, W - tile) + 1, stride))
    if ys[-1] + tile < H: ys.append(H - tile)
    if xs[-1] + tile < W: xs.append(W - tile)
    batch, offs = [], []
    def run():
        for r, (ox, oy) in zip(model.predict(batch, imgsz=tile, conf=conf, verbose=False), offs):
            # 30/09: modelos con clase auxiliar de negativos (ds23): EVAL_CLASES_FUERA = ids de clase a descartar
            # (YOLO ds23: '1'; RF-DETR ds23: '2', por la categoria 0 'padre' de COCO). Vacio = se usan todas.
            cl = r.boxes.cls.tolist() if getattr(r.boxes, 'cls', None) is not None else [0] * len(r.boxes.conf)
            for b, c, k in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), cl):
                if int(k) in FUERA: continue
                dets.append([b[0]+ox, b[1]+oy, b[2]+ox, b[3]+oy, c])
        batch.clear(); offs.clear()
    for y in ys:
        for x in xs:
            t = img[max(0,y):y+tile, max(0,x):x+tile]
            if t.shape != (tile, tile): t = cv2.copyMakeBorder(t, 0, tile-t.shape[0], 0, tile-t.shape[1], cv2.BORDER_CONSTANT, value=255)
            if (t < 128).sum() == 0: continue
            batch.append(cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)); offs.append((max(0,x), max(0,y)))
            if len(batch) == int(os.environ.get('EVAL_LOTE', '32')): run()   # 28/09: EVAL_LOTE si la GPU esta compartida
    if batch: run()
    # NMS + supresion de anidadas
    dets.sort(key=lambda d: -d[4]); keep = []
    for d in dets:
        ok = True
        for k in keep:
            ix = max(0, min(d[2], k[2]) - max(d[0], k[0])); iy = max(0, min(d[3], k[3]) - max(d[1], k[1])); I = ix * iy
            A = (d[2]-d[0])*(d[3]-d[1]); B = (k[2]-k[0])*(k[3]-k[1])
            if I / (A + B - I + 1e-9) > .45 or (I / (min(A, B) + 1e-9) > .7 and max(A, B) < 4 * min(A, B)): ok = False; break
        if ok: keep.append(d)
    return keep

SCALES = [float(s) for s in os.environ.get('EVAL_SCALES', '1.0,1.6').split(',')]
def fuse(D, iou=.45, ios=.7, ar=4.0):
    D = sorted(D, key=lambda d: -d[4]); keep = []
    for d in D:
        ok = True
        for k in keep:
            ix = max(0, min(d[2], k[2]) - max(d[0], k[0])); iy = max(0, min(d[3], k[3]) - max(d[1], k[1])); I = ix * iy
            A = (d[2]-d[0])*(d[3]-d[1]); B = (k[2]-k[0])*(k[3]-k[1])
            if I / (A + B - I + 1e-12) > iou or (I / (min(A, B) + 1e-12) > ios and max(A, B) < ar * min(A, B)): ok = False; break
        if ok: keep.append(d)
    return keep

def match(gt, dets_cad):
    """Asignacion optima (Hungaro): cada GT a lo sumo una deteccion valida, minimizando distancia."""
    from scipy.optimize import linear_sum_assignment
    nG, nD = len(gt), len(dets_cad)
    if nG == 0 or nD == 0: return [], list(range(nG)), list(range(nD))
    BIG = 1e6; C = np.full((nG, nD), BIG)
    for gi, g in enumerate(gt):
        for j, d in enumerate(dets_cad):
            x0, y0, x1, y1 = d[:4]; mx = (x1-x0)*.2; my = (y1-y0)*.2
            inside = x0-mx <= g['c'][0] <= x1+mx and y0-my <= g['c'][1] <= y1+my
            if g['b'] is not None:
                cx, cy = (x0+x1)/2, (y0+y1)/2; b = g['b']
                inside = inside or (b[0] <= cx <= b[2] and b[1] <= cy <= b[3])
            if inside: C[gi, j] = np.hypot((x0+x1)/2 - g['c'][0], (y0+y1)/2 - g['c'][1])
    r, c = linear_sum_assignment(C)
    tp = [(gi, j) for gi, j in zip(r, c) if C[gi, j] < BIG]
    mg = {gi for gi, _ in tp}; md = {j for _, j in tp}
    return tp, [gi for gi in range(nG) if gi not in mg], [j for j in range(nD) if j not in md]

def metrics_at(gt, dc, th):
    d = [x for x in dc if x[4] >= th]
    tp, fn, fp = match(gt, d)
    return dict(th=th, recall=round(len(tp)/max(1, len(gt)), 4), precision=round(len(tp)/max(1, len(d)), 4), tp=len(tp), fn=len(fn), fp=len(fp))

if __name__ == '__main__':
    w = sys.argv[1] if len(sys.argv) > 1 else os.path.join(PACK, 'best_componente_v2.pt')
    conf = float(sys.argv[2]) if len(sys.argv) > 2 else 0.20
    tag = sys.argv[3] if len(sys.argv) > 3 else 'eval'
    model = RFDETRComoYOLO(w) if w.endswith('.pth') else YOLO(w); out = os.path.join(WORK, 'eval', tag); os.makedirs(out, exist_ok=True); resumen = {}
    for name, dxf, gtp in [q for q in PLANOS if not os.environ.get("EVAL_SOLO") or q[0] == os.environ["EVAL_SOLO"]]:   # 30/09: EVAL_SOLO=<plano>
        doc = ezdxf.readfile(os.path.join(BASE, dxf)); ppc = auto_ppc(doc)
        dc = []; dets = []
        for k, sc in enumerate(SCALES):   # multi-escala: el render a 1.6x agranda simbolos chicos (borneras) -> mucho mas recall
            img_k, meta_k = render_doc(doc, ppc * sc); img_k = darken(img_k)
            dk = infer(model, img_k, 0.05)
            if k == 0: img, meta, dets = img_k, meta_k, dk
            for d in dk:
                xa, ya = px2cad(meta_k, d[0], d[3]); xb, yb = px2cad(meta_k, d[2], d[1]); dc.append([xa, ya, xb, yb, d[4]])
        if len(SCALES) > 1: dc = fuse(dc)
        dets = [[*cad2px(meta, d[0], d[3]), *cad2px(meta, d[2], d[1]), d[4]] for d in dc]
        gt = load_gt(name, os.path.join(BASE, gtp))
        tabla = [metrics_at(gt, dc, t) for t in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)]
        # umbral maximo con recall 100%
        tp_all, fn_all, _ = match(gt, dc)
        th100 = min([dc[j][4] for _, j in tp_all], default=0) if not fn_all else None
        sel_d = [x for x in dc if x[4] >= conf]; sel_i = [i for i, x in enumerate(dc) if x[4] >= conf]
        tp, fn, fp = match(gt, sel_d)
        resumen[name] = dict(ppc=round(ppc, 2), gt=len(gt), conf=conf, **{k: v for k, v in metrics_at(gt, dc, conf).items() if k != 'th'},
                             umbral_max_recall100=th100, fn_bloques=[gt[i]['n'] for i in fn], por_umbral=tabla)
        print(name, {k: resumen[name][k] for k in ('gt', 'recall', 'precision', 'tp', 'fn', 'fp', 'umbral_max_recall100')})
        v = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR); D = [dets[i] for i in sel_i]
        for _, j in tp: d = D[j]; cv2.rectangle(v, (int(d[0]), int(d[1])), (int(d[2]), int(d[3])), (0, 170, 0), 2); cv2.putText(v, f'{d[4]:.2f}', (int(d[0]), int(d[1])-3), 0, .4, (0,130,0), 1)
        for j in fp: d = D[j]; cv2.rectangle(v, (int(d[0]), int(d[1])), (int(d[2]), int(d[3])), (255, 0, 0), 2); cv2.putText(v, f'{d[4]:.2f}', (int(d[0]), int(d[1])-3), 0, .4, (255,0,0), 1)
        for i in fn:
            x, y = cad2px(meta, *gt[i]['c']); cv2.circle(v, (int(x), int(y)), 25, (0, 0, 255), 3)
        cv2.imwrite(os.path.join(out, f'{name}_visual.png'), v)
        with open(os.path.join(out, f'{name}_detecciones.csv'), 'w', newline='') as f:
            wr = csv.writer(f); wr.writerow(['x1', 'y1', 'x2', 'y2', 'conf'])
            for d in dc: wr.writerow([*[round(v_, 4) for v_ in d[:4]], round(d[4], 3)])
    json.dump(resumen, open(os.path.join(out, 'resumen.json'), 'w'), indent=1, ensure_ascii=False)

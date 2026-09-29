"""Mide la union de dos modelos: recall que se consigue juntando sus detecciones.

Los planes R y U fallan en componentes DISTINTOS -R solo pierde el `PLC` en marco punteado, U
solo `C7AFF483F` y un amperimetro-, asi que la union de los dos puede no perder ninguno. Para un
proyecto cuya prioridad declarada es recall 100% aceptando falsos positivos, eso es un resultado
utilizable: se corren los dos modelos y se juntan las cajas.

    GT_V4=1 py -3 src/ensamble.py V4_v13_R V4_v16_U
"""
import sys, os, csv, importlib
sys.path.insert(0, os.path.dirname(__file__))
from paths import BASE, WORK
import evaluate as ev
import rescore as rs

TH = (0.05, 0.25)


def main():
    tags = sys.argv[1:] or ['V4_v13_R', 'V4_v16_U']
    for th in TH:
        gt_n = tp_t = fn_t = fp_t = 0
        detalle = []
        for suf, qet in (('005', False), ('marcelo', True)):
            if qet:
                os.environ['EVAL_SET'] = 'qet'; os.environ['QET_DIR'] = 'test_marcelo'
            else:
                os.environ.pop('EVAL_SET', None)
            importlib.reload(ev)
            for pl, dxf, gtp in ev.PLANOS:
                dc = []
                for t in tags:
                    f = os.path.join(WORK, 'eval', '%s_%s' % (t, suf), pl + '_detecciones.csv')
                    if not os.path.exists(f):
                        continue
                    dc += [[float(r['x1']), float(r['y1']), float(r['x2']), float(r['y2']), float(r['conf'])]
                           for r in csv.DictReader(open(f)) if float(r['conf']) >= th]
                if not dc:
                    continue
                dc = ev.fuse(dc) if hasattr(ev, 'fuse') else dc
                gt = ev.load_gt(pl, os.path.join(BASE, gtp))
                tp, fn, fp = rs.match_rapido(gt, dc)
                gt_n += len(gt); tp_t += len(tp); fn_t += len(fn); fp_t += len(fp)
                if fn:
                    detalle += ['%s:%s' % (pl[:14], gt[i]['n'].split('$')[-1][:18]) for i in fn]
        print('union %s @%.2f -> GT=%d  FN=%d  recall=%.4f  FP=%d  prec=%.4f'
              % ('+'.join(tags), th, gt_n, fn_t, tp_t / max(1, gt_n), fp_t, tp_t / max(1, tp_t + fp_t)))
        if detalle:
            print('   FN:', ', '.join(detalle))


if __name__ == '__main__':
    main()

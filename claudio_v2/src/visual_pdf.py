"""PDF vectorial de una evaluacion: el DXF dibujado + detecciones a un umbral, con zoom sin perder calidad.

25/09, pedido de Tomas (para mostrarle a un amigo las detecciones de RF-DETR en los planos de test).
Los `_visual.png` de evaluate.py son raster y estan al umbral interno 0,05 (con todos los FP). Aca:
  - el DXF se dibuja con la misma configuracion de render que la evaluacion (render.CFG), pero en un
    PDF vectorial;
  - las detecciones (CSV de `work/eval/<tag>`, coordenadas CAD) se filtran a --conf y se emparejan con
    el GT igual que siempre (rescore.match_rapido): verde = acierto, azul = falso positivo, circulo
    rojo = componente perdido. Las cajas van en coordenadas CAD sobre los mismos ejes: posicion exacta.

    GT_V4=1 py -3 src/visual_pdf.py V4_RF_nano --conf 0.13 --salida work/visual_RF
"""
import sys, os, csv, argparse, importlib
sys.path.insert(0, os.path.dirname(__file__))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import ezdxf
from ezdxf.addons.drawing import RenderContext, Frontend
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend
from paths import BASE, WORK
import render
import evaluate as ev
import rescore as rs


def plano_pdf(dxf, gt, dets, salida, titulo):
    doc = ezdxf.readfile(dxf)
    x0, y0, x1, y1 = render.plan_extents(doc)
    ancho_mm = 1189.                                  # A0 de ancho: se lee entero y se hace zoom
    alto_mm = ancho_mm * (y1 - y0) / max(1e-9, x1 - x0)
    fig = plt.figure(figsize=(ancho_mm / 25.4, alto_mm / 25.4 + .6))
    ax = fig.add_axes([0, 0, 1, alto_mm / (alto_mm + .6 * 25.4)])
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect('equal'); ax.axis('off')
    Frontend(RenderContext(doc), MatplotlibBackend(ax), config=render.CFG).draw_layout(doc.modelspace(), finalize=False)
    tp, fn, fp = rs.match_rapido(gt, [list(d) for d in dets])
    lw = .35
    for _, j in tp:
        d = dets[j]; ax.add_patch(Rectangle((d[0], d[1]), d[2] - d[0], d[3] - d[1], fill=False, ec='#00a000', lw=lw, zorder=5))
    for j in fp:
        d = dets[j]; ax.add_patch(Rectangle((d[0], d[1]), d[2] - d[0], d[3] - d[1], fill=False, ec='#1060ff', lw=lw, zorder=5))
    r = .012 * max(x1 - x0, y1 - y0)
    for i in fn:
        ax.add_patch(Circle(gt[i]['c'], r, fill=False, ec='red', lw=1.2, zorder=6))
    fig.text(.005, .995, '%s   |   verde: acierto (%d)   azul: falso positivo (%d)   circulo rojo: perdido (%d)   |   %d componentes'
             % (titulo, len(tp), len(fp), len(fn), len(gt)), va='top', fontsize=14)
    fig.savefig(salida)
    plt.close(fig)
    return len(tp), len(fp), len(fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('tag'); ap.add_argument('--conf', type=float, default=.13)
    ap.add_argument('--salida', default=os.path.join(WORK, 'visual'))
    a = ap.parse_args()
    os.makedirs(a.salida, exist_ok=True)
    for suf, qet in (('005', False), ('marcelo', True)):
        if qet:
            os.environ['EVAL_SET'] = 'qet'; os.environ['QET_DIR'] = 'test_marcelo'
        else:
            os.environ.pop('EVAL_SET', None)
        importlib.reload(ev)
        for pl, dxf, gtp in ev.PLANOS:
            f = os.path.join(WORK, 'eval', '%s_%s' % (a.tag, suf), pl + '_detecciones.csv')
            if not os.path.exists(f):
                print('falta', f); continue
            dets = [[float(r['x1']), float(r['y1']), float(r['x2']), float(r['y2']), float(r['conf'])]
                    for r in csv.DictReader(open(f)) if float(r['conf']) >= a.conf]
            gt = ev.load_gt(pl, os.path.join(BASE, gtp))
            out = os.path.join(a.salida, '%s_conf%03d.pdf' % (pl, round(a.conf * 100)))
            n = plano_pdf(os.path.join(BASE, dxf), gt, dets, out, '%s  -  RF-DETR Nano  -  conf %.2f' % (pl, a.conf))
            print('%s: aciertos %d, FP %d, perdidos %d -> %s' % (pl, *n, out))


if __name__ == '__main__':
    main()

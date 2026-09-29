"""Separacion de confianza entre verdaderos y falsos positivos.

El objetivo del proyecto no es solo recall 100%: ademas se quiere que los aciertos salgan con
confianza ALTA y los falsos positivos con confianza BAJA, para que la etapa 2 pueda filtrar por
umbral sin perder nada. Esta metrica mide exactamente eso.

  - `recall@conf`: hasta que umbral se puede subir sin perder ni un componente. Es el numero
    que mas importa: si es alto, el modelo se puede usar con un corte agresivo.
  - mediana de confianza de TP y de FP, y cuanto se separan.
  - `FP por encima del TP mas bajo`: cuantos falsos positivos son mas confiados que el acierto
    menos confiado. Son los que no se pueden filtrar por umbral sin perder un componente.

    GT_V4=1 py -3 src/metrica_confianza.py V4_v13_R V3_v9_N
"""
import sys, os, csv, importlib
sys.path.insert(0, os.path.dirname(__file__))
from paths import BASE, WORK
import evaluate as ev
import rescore as rs


def main():
    prefs = sys.argv[1:] or ['V4_v13_R']
    print('%-10s %6s %5s | %9s %9s %7s | %s' %
          ('modelo', 'TP', 'FN', 'conf TP', 'conf FP', 'sep', 'umbral con recall 100% / FP arriba del peor TP'))
    for pref in prefs:
        tp_c, fp_c = [], []
        n_gt = n_fn = 0
        for suf, qet in (('005', False), ('marcelo', True)):
            if qet:
                os.environ['EVAL_SET'] = 'qet'; os.environ['QET_DIR'] = 'test_marcelo'
            else:
                os.environ.pop('EVAL_SET', None)
            importlib.reload(ev)
            for pl, dxf, gtp in ev.PLANOS:
                f = os.path.join(WORK, 'eval', '%s_%s' % (pref, suf), pl + '_detecciones.csv')
                if not os.path.exists(f):
                    continue
                gt = ev.load_gt(pl, os.path.join(BASE, gtp))
                dc = [[float(r['x1']), float(r['y1']), float(r['x2']), float(r['y2']), float(r['conf'])]
                      for r in csv.DictReader(open(f))]
                d = [x for x in dc if x[4] >= 0.05]
                tp, fn, fp = rs.match_rapido(gt, d)
                n_gt += len(gt); n_fn += len(fn)
                for _g, j in tp:
                    tp_c.append(d[j][4])
                for j in fp:
                    fp_c.append(d[j][4])
        if not tp_c:
            print('%-10s sin datos' % pref); continue
        tp_c.sort(); fp_c.sort()
        med_tp = tp_c[len(tp_c) // 2]; med_fp = fp_c[len(fp_c) // 2] if fp_c else 0
        peor_tp = tp_c[0]
        fp_arriba = sum(1 for c in fp_c if c >= peor_tp)
        # umbral al que todavia no se pierde ningun componente
        umbral = peor_tp if n_fn == 0 else 0.0
        print('%-10s %6d %5d | %9.3f %9.3f %7.3f | %.3f  /  %d de %d FP (%.1f%%)' %
              (pref, len(tp_c), n_fn, med_tp, med_fp, med_tp - med_fp,
               umbral, fp_arriba, len(fp_c), 100. * fp_arriba / max(1, len(fp_c))))


if __name__ == '__main__':
    main()

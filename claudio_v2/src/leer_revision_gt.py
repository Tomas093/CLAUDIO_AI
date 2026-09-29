# -*- coding: utf-8 -*-
"""Lee la auditoria de la VERDAD DE TERRENO y dice si se puede confiar en ella.

  py -3 src/leer_revision_gt.py [carpeta_revision_gt]

Lo que importa de esta corrida no es cuanto mejora el modelo sino si la regla con la que se
mide sirve. Dos preguntas:

  - Tarea C: cuantas cajas del GT encierran algo que no es un componente, o mas de uno. Cada
    una de esas es una fila del GT que le esta pidiendo al modelo algo equivocado.
  - Tarea D: cuantos componentes le FALTAN al GT. Esos no se le reclaman a ningun modelo, asi
    que el recall medido es optimista: el denominador esta incompleto.

El segundo numero es el que puede invalidar las comparaciones entre modelos, porque si un
modelo detecta un componente que el GT no tiene, hoy cuenta como falso positivo.
"""
import os, sys, json, collections

PACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(PACK), 'revision_gt')


def jsonl(p):
    out = []
    if not os.path.exists(p):
        return out
    for ln in open(p, encoding='utf-8'):
        ln = ln.strip().strip(',')
        if not ln or ln.startswith('```') or ln in ('[', ']'):
            continue
        try:
            o = json.loads(ln)
        except ValueError:
            continue
        if isinstance(o, dict) and o.get('id'):
            out.append(o)
    return out


def main():
    man_c = {m['id']: m for m in jsonl(os.path.join(DIR, 'manifest_c.jsonl'))}
    man_d = {m['id']: m for m in jsonl(os.path.join(DIR, 'manifest_d.jsonl'))}
    rc = [r for r in jsonl(os.path.join(DIR, 'respuestas_c.jsonl')) if r['id'] in man_c]
    rd = [r for r in jsonl(os.path.join(DIR, 'respuestas_d.jsonl')) if r['id'] in man_d]
    print('manifest C %d | respuestas C %d' % (len(man_c), len(rc)))
    print('manifest D %d | respuestas D %d' % (len(man_d), len(rd)))

    if rc:
        dentro = collections.Counter()
        ajuste = collections.Counter()
        por_plano = collections.defaultdict(collections.Counter)
        por_bloque = collections.defaultdict(collections.Counter)
        malas = []
        for r in rc:
            try:
                n = int(r.get('dentro', -1))
            except (TypeError, ValueError):
                n = -1
            m = man_c[r['id']]
            dentro[n] += 1
            ajuste[str(r.get('ajuste', '?'))] += 1
            est = 'ok' if n == 1 else ('vacia' if n == 0 else 'varios')
            por_plano[m['plano']][est] += 1
            por_bloque[m.get('bloque', '?')][est] += 1
            if n != 1:
                malas.append({'id': r['id'], 'plano': m['plano'], 'idx': m['idx'],
                              'bloque': m.get('bloque', '?'), 'dentro': n,
                              'tipo': r.get('tipo', ''), 'caja': m.get('caja')})
        tot = len(rc)
        print('\nTAREA C -- que encierra cada caja del GT')
        print('  ' + '-' * 58)
        print('  %-40s %6d %5.1f%%' % ('0  no hay componente', dentro[0], 100.*dentro[0]/tot))
        print('  %-40s %6d %5.1f%%' % ('1  correcta', dentro[1], 100.*dentro[1]/tot))
        v = sum(x for k, x in dentro.items() if k >= 2)
        print('  %-40s %6d %5.1f%%' % ('2+ abarca varios', v, 100.*v/tot))
        print('\nTAREA C -- encuadre del GT')
        print('  ' + '-' * 58)
        for k, x in sorted(ajuste.items(), key=lambda z: -z[1]):
            print('  %-40s %6d %5.1f%%' % (k, x, 100.*x/tot))
        print('\nTAREA C -- por plano')
        print('  ' + '-' * 58)
        print('  %-24s %6s %7s %7s' % ('plano', 'ok', 'vacia', 'varios'))
        for p in sorted(por_plano):
            c = por_plano[p]
            print('  %-24s %6d %7d %7d' % (p[:24], c['ok'], c['vacia'], c['varios']))
        sosp = [(b, c) for b, c in por_bloque.items() if c['vacia'] + c['varios'] >= 2]
        if sosp:
            print('\n  bloques del GT con 2+ cajas dudosas (candidatos a revisar enteros):')
            for b, c in sorted(sosp, key=lambda z: -(z[1]['vacia'] + z[1]['varios']))[:12]:
                print('    %-40s ok %3d  vacia %3d  varios %3d' % (b[:40], c['ok'], c['vacia'], c['varios']))
        with open(os.path.join(DIR, 'gt_dudoso.jsonl'), 'w', encoding='utf-8') as fh:
            for x in malas:
                fh.write(json.dumps(x, ensure_ascii=False) + '\n')
        print('\n  %d cajas de GT dudosas -> revision_gt/gt_dudoso.jsonl' % len(malas))

    if rd:
        seg = collections.Counter()
        tipos = collections.Counter()
        por_plano = collections.defaultdict(lambda: [0, 0])
        falt = []
        for r in rd:
            m = man_d[r['id']]
            por_plano[m['plano']][1] += m.get('cajas_dibujadas', 0)
            for f in (r.get('faltantes') or []):
                if not isinstance(f, dict):
                    continue
                s = str(f.get('seguridad', 'media')).lower()
                seg[s] += 1
                tipos[str(f.get('tipo', '?')).lower()] += 1
                if s in ('alta', 'media'):
                    por_plano[m['plano']][0] += 1
                try:
                    e = float(m.get('escala', 1.0)) or 1.0
                    X = m['origen'][0] + float(f['x']) / e
                    Y = m['origen'][1] + float(f['y']) / e
                except (KeyError, TypeError, ValueError):
                    continue
                falt.append({'plano': m['plano'], 'x_px': round(X, 1), 'y_px': round(Y, 1),
                             'tipo': f.get('tipo', '?'), 'seguridad': s, 'celda': r['id']})
        print('\nTAREA D -- componentes que le FALTAN al GT')
        print('  ' + '-' * 58)
        for k, x in sorted(seg.items(), key=lambda z: -z[1]):
            print('  %-40s %6d' % (k, x))
        print('\nTAREA D -- de que tipo')
        print('  ' + '-' * 58)
        for k, x in tipos.most_common(12):
            print('  %-40s %6d' % (k, x))
        print('\nTAREA D -- cuanto se infla el recall por plano')
        print('  ' + '-' * 58)
        print('  %-24s %8s %8s %s' % ('plano', 'gt_visto', 'faltan', 'GT incompleto en'))
        for p in sorted(por_plano, key=lambda z: -por_plano[z][0]):
            f_, d_ = por_plano[p]
            pct = (100.0 * f_ / (d_ + f_)) if (d_ + f_) else 0
            print('  %-24s %8d %8d %6.1f%%' % (p[:24], d_, f_, pct))
        tf = sum(v[0] for v in por_plano.values())
        td = sum(v[1] for v in por_plano.values())
        if td + tf:
            print('\n  En la muestra auditada, al GT le falta el %.1f%% de los componentes.' % (100.0*tf/(td+tf)))
            print('  Si se confirma, el recall real es MAS BAJO que el medido: lo que el GT no')
            print('  tiene no se le reclama a ningun modelo, y ademas cuenta como falso positivo')
            print('  cuando un modelo si lo detecta.')
        with open(os.path.join(DIR, 'gt_faltantes.jsonl'), 'w', encoding='utf-8') as fh:
            for f in falt:
                fh.write(json.dumps(f, ensure_ascii=False) + '\n')
        print('\n  %d faltantes del GT -> revision_gt/gt_faltantes.jsonl' % len(falt))

    if not rc and not rd:
        print('\nNo hay respuestas todavia. Se esperan en:')
        print('  %s' % os.path.join(DIR, 'respuestas_c.jsonl'))
        print('  %s' % os.path.join(DIR, 'respuestas_d.jsonl'))


if __name__ == '__main__':
    main()

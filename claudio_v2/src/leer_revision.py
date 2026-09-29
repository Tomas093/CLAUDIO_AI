# -*- coding: utf-8 -*-
"""Lee las respuestas del revisor de vision y las cruza con los manifest.

Salida: los numeros que no se podian medir sin verdad de terreno sobre los PDF --
cuantas cajas abarcan mas de un componente, cuantas caen sobre texto, y cuantos
componentes se perdio el detector.

  py -3 src/leer_revision.py [carpeta_revision_ia]

Lo que el revisor dice es un dato ruidoso, no un oraculo: por eso los faltantes se
reportan separados por nivel de seguridad y los de seguridad baja no se suman al total.
"""
import os, sys, json, collections

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(RAIZ, 'revision_ia')


def jsonl(p):
    """Tolera lineas sueltas rotas y el markdown que a veces se cuela."""
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


def tabla(titulo, filas, tot):
    print('\n%s' % titulo)
    print('  ' + '-' * 58)
    for k, v in filas:
        pct = (' %5.1f%%' % (100.0 * v / tot)) if tot else ''
        print('  %-40s %6d%s' % (k, v, pct))


def main():
    man_a = {m['id']: m for m in jsonl(os.path.join(DIR, 'manifest_a.jsonl'))}
    man_b = {m['id']: m for m in jsonl(os.path.join(DIR, 'manifest_b.jsonl'))}
    ra = jsonl(os.path.join(DIR, 'respuestas_a.jsonl'))
    rb = jsonl(os.path.join(DIR, 'respuestas_b.jsonl'))
    print('manifest A %d | respuestas A %d' % (len(man_a), len(ra)))
    print('manifest B %d | respuestas B %d' % (len(man_b), len(rb)))

    # ---------------- tarea A
    ra = [r for r in ra if r['id'] in man_a]
    if ra:
        dentro = collections.Counter()
        ajuste = collections.Counter()
        por_motivo = collections.defaultdict(collections.Counter)
        malas = []
        for r in ra:
            try:
                n = int(r.get('dentro', -1))
            except (TypeError, ValueError):
                n = -1
            dentro[n] += 1
            ajuste[str(r.get('ajuste', '?'))] += 1
            mot = man_a[r['id']].get('motivo', '?')
            por_motivo[mot][('ok' if n == 1 else ('vacia' if n == 0 else 'varios'))] += 1
            if n != 1:
                malas.append((r['id'], man_a[r['id']]['plano'], n, r.get('tipo', '')))
        tot = len(ra)
        tabla('TAREA A -- cuantos componentes encierra cada caja', [
            ('0  la caja no tiene componente', dentro[0]),
            ('1  correcta', dentro[1]),
            ('2+ abarca varios componentes', sum(v for k, v in dentro.items() if k >= 2)),
            ('   (sin respuesta valida)', dentro[-1]),
        ], tot)
        tabla('TAREA A -- encuadre', sorted(ajuste.items(), key=lambda x: -x[1]), tot)

        print('\nTAREA A -- por que se habia marcado como sospechosa')
        print('  ' + '-' * 58)
        print('  %-14s %6s %7s %7s' % ('motivo', 'ok', 'vacia', 'varios'))
        for mot in sorted(por_motivo):
            c = por_motivo[mot]
            print('  %-14s %6d %7d %7d' % (mot, c['ok'], c['vacia'], c['varios']))
        # la tasa en 'muestra' es la tasa base: dice cuanto pasa esto en una caja cualquiera
        m = por_motivo.get('muestra')
        if m and sum(m.values()):
            n = sum(m.values())
            print('\n  tasa base (cajas al azar): %.1f%% correctas, %.1f%% con varios, %.1f%% vacias'
                  % (100.*m['ok']/n, 100.*m['varios']/n, 100.*m['vacia']/n))

        with open(os.path.join(DIR, 'cajas_malas.jsonl'), 'w', encoding='utf-8') as fh:
            for i, p, n, t in malas:
                fh.write(json.dumps({'id': i, 'plano': p, 'dentro': n, 'tipo': t},
                                    ensure_ascii=False) + '\n')
        print('\n  %d cajas no-correctas -> revision_ia/cajas_malas.jsonl' % len(malas))

    # ---------------- tarea B
    rb = [r for r in rb if r['id'] in man_b]
    if rb:
        seg = collections.Counter()
        tipos = collections.Counter()
        por_plano = collections.defaultdict(lambda: [0, 0])   # [faltantes alta+media, dibujadas]
        falt = []
        for r in rb:
            m = man_b[r['id']]
            por_plano[m['plano']][1] += m.get('cajas_dibujadas', 0)
            for f in (r.get('faltantes') or []):
                if not isinstance(f, dict):
                    continue
                s = str(f.get('seguridad', 'media')).lower()
                seg[s] += 1
                tipos[str(f.get('tipo', '?')).lower()] += 1
                if s in ('alta', 'media'):
                    por_plano[m['plano']][0] += 1
                # a coordenadas del plano: se deshace la reduccion y se suma el origen
                try:
                    e = float(m.get('escala', 1.0)) or 1.0
                    X = m['origen'][0] + float(f['x']) / e
                    Y = m['origen'][1] + float(f['y']) / e
                except (KeyError, TypeError, ValueError):
                    continue
                falt.append({'plano': m['plano'], 'x': round(X, 1), 'y': round(Y, 1),
                             'tipo': f.get('tipo', '?'), 'seguridad': s, 'celda': r['id']})
        tabla('TAREA B -- componentes que el detector se perdio',
              sorted(seg.items(), key=lambda x: -x[1]), sum(seg.values()))
        tabla('TAREA B -- de que tipo', tipos.most_common(12), sum(tipos.values()))

        print('\nTAREA B -- por plano (faltantes de seguridad alta o media)')
        print('  ' + '-' * 58)
        print('  %-40s %7s %8s' % ('plano', 'faltan', 'marcadas'))
        for p in sorted(por_plano, key=lambda x: -por_plano[x][0]):
            f_, d_ = por_plano[p]
            print('  %-40s %7d %8d' % (p[:40], f_, d_))
        tf = sum(v[0] for v in por_plano.values())
        td = sum(v[1] for v in por_plano.values())
        if td:
            print('\n  recall aparente: %.2f%%  (%d de %d)' % (100.0*td/(td+tf), td, td+tf))
            print('  ojo: las celdas se solapan 12%, asi que hay cajas contadas dos veces.')
            print('  El numero sirve para comparar modelos entre si, no como valor absoluto.')

        with open(os.path.join(DIR, 'faltantes.jsonl'), 'w', encoding='utf-8') as fh:
            for f in falt:
                fh.write(json.dumps(f, ensure_ascii=False) + '\n')
        print('\n  %d faltantes con coordenadas -> revision_ia/faltantes.jsonl' % len(falt))

    if not ra and not rb:
        print('\nNo hay respuestas todavia. Se esperan en:')
        print('  %s' % os.path.join(DIR, 'respuestas_a.jsonl'))
        print('  %s' % os.path.join(DIR, 'respuestas_b.jsonl'))


if __name__ == '__main__':
    main()

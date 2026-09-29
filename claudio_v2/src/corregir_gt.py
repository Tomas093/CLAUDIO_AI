# -*- coding: utf-8 -*-
"""Agrega al GT los componentes que la auditoria encontro que le faltaban.

La auditoria de `revision_gt` (tarea D) dejo en `gt_faltantes.jsonl` los componentes sin caja,
con coordenadas en PIXELES del render. Aca se pasan a coordenadas CAD y se escriben csv nuevos.

Por que importa: lo que el GT no tiene no se le reclama a ningun modelo, asi que el recall
medido es optimista; y peor, cuando un modelo SI detecta uno de esos, hoy se le cuenta como
falso positivo. Sobre LU-UN-01 y nyw-un-01 al GT le falta el 17% de los componentes, asi que
la precision de ~50% que venimos midiendo esta subestimada.

**Los csv originales NO se tocan.** Se escribe `<nombre>_v2.csv` al lado, y se mueve una copia
del original a `_para_borrar/`. Cambiar el GT cambia la vara con la que se compara todo, asi
que conviene medir con los dos y ver la diferencia antes de adoptarlo.

  py -3 src/corregir_gt.py [--aplicar]

Sin `--aplicar` solo dice que haria.
"""
import os, sys, csv, json, shutil, collections

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import BASE
from render import render_doc, px2cad
from scale import auto_ppc
import ezdxf

DIR = os.path.join(BASE, 'revision_gt')

# 20/09, confirmado por Tomas: las termomagneticas de las posiciones de RESERVA si son
# componentes ("las de reserva si son un componente por las dudas"). Por el mismo criterio, y
# porque la regla del proyecto es recall 100% con falsos positivos aceptados, se incluyen
# tambien las bobinas de apertura. Si hay que sacarlas, alcanza con ponerlas aca.
TIPOS_FUERA = set(t.strip() for t in os.environ.get('GT_TIPOS_FUERA', '').split(',') if t.strip())

# la seguridad 'baja' del revisor no entra: son corazonadas suyas, y una caja de GT inventada
# es peor que una que falta, porque mide mal a todos los modelos a la vez
SEG_OK = {'alta', 'media'}

PLANOS = {
    'test1': ('test1.dxf', 'test/test_1/verdad_terreno/test1_completo.csv'),
    'test_2': ('test_2.dxf', 'test/test_2/verdad_terreno/test_2_completo.csv'),
    'fl_un_02': ('dxf/FL-UN-02_tablero_1.dxf', 'dxf/fl_un_02_gt_completo.csv'),
    'tsss_2': ('TSSS_2 (1).dxf', 'dxf/tsss_2_gt_completo.csv'),
    'EZE4077': ('claudio_v2/data/test_marcelo/EZE4077-IE-UF001-02.dxf',
                'claudio_v2/data/test_marcelo/EZE4077-IE-UF001-02_gt.csv'),
    'LU-UN-01': ('claudio_v2/data/test_marcelo/LU-UN-01 Esquemas Unifilares.dxf',
                 'claudio_v2/data/test_marcelo/LU-UN-01 Esquemas Unifilares_gt.csv'),
    'nyw-un-01': ('claudio_v2/data/test_marcelo/nyw-un-01 esquemas unifilares.dxf',
                  'claudio_v2/data/test_marcelo/nyw-un-01 esquemas unifilares_gt.csv'),
}


def lado_tipico(filas):
    """Lado medio de las cajas del GT de ese plano, en unidades CAD.

    El revisor da el CENTRO del componente que falta, no su caja. Se le arma una caja del
    tamano tipico del plano: no es exacta, pero el evaluador asigna por distancia al centro,
    asi que lo que importa es que el centro este bien.
    """
    ws, hs = [], []
    for r in filas:
        if r.get('x1'):
            ws.append(abs(float(r['x2']) - float(r['x1'])))
            hs.append(abs(float(r['y2']) - float(r['y1'])))
    if not ws:
        return 0.3, 0.3
    ws.sort(); hs.sort()
    return ws[len(ws)//2], hs[len(hs)//2]


def main():
    aplicar = '--aplicar' in sys.argv
    falt = collections.defaultdict(list)
    for ln in open(os.path.join(DIR, 'gt_faltantes.jsonl'), encoding='utf-8'):
        o = json.loads(ln)
        if o.get('seguridad') not in SEG_OK:
            continue
        if str(o.get('tipo', '')).lower() in TIPOS_FUERA:
            continue
        falt[o['plano']].append(o)
    print('faltantes a agregar (seguridad alta/media%s):' %
          (', sin %s' % ','.join(sorted(TIPOS_FUERA)) if TIPOS_FUERA else ''))
    for p in sorted(falt):
        print('   %-14s %4d' % (p, len(falt[p])))
    print('   %-14s %4d' % ('TOTAL', sum(len(v) for v in falt.values())))
    if not aplicar:
        print('\n(sin --aplicar no se escribe nada)')
        return

    backup = os.path.join(BASE, '_para_borrar', 'gt_antes_de_auditoria_20092026')
    os.makedirs(backup, exist_ok=True)
    for nom, fs in sorted(falt.items()):
        if nom not in PLANOS:
            print('[aviso] plano desconocido: %s' % nom); continue
        dxf, gtp = PLANOS[nom]
        fd, fg = os.path.join(BASE, dxf), os.path.join(BASE, gtp)
        if not (os.path.exists(fd) and os.path.exists(fg)):
            print('[aviso] falta %s' % nom); continue
        doc = ezdxf.readfile(fd)
        ppc = auto_ppc(doc)
        _img, meta = render_doc(doc, ppc)
        with open(fg, newline='', encoding='utf-8', errors='ignore') as f:
            rd = csv.DictReader(f)
            campos = list(rd.fieldnames or [])
            filas = list(rd)
        w2, h2 = [v / 2. for v in lado_tipico(filas)]
        nuevas = 0
        for o in fs:
            X, Y = px2cad(meta, float(o['x_px']), float(o['y_px']))
            fila = {c: '' for c in campos}
            for c, v in (('x_cad', X), ('y_cad', Y), ('x1', X - w2), ('y1', Y - h2),
                         ('x2', X + w2), ('y2', Y + h2)):
                if c in fila:
                    fila[c] = '%.4f' % v
            if 'block_name' in fila:
                fila['block_name'] = 'AUDIT-%s' % str(o.get('tipo', '?')).replace(' ', '_')
            filas.append(fila)
            nuevas += 1
        shutil.copy2(fg, os.path.join(backup, os.path.basename(fg)))
        sal = fg[:-4] + '_v2.csv'
        with open(sal, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=campos)
            w.writeheader()
            w.writerows(filas)
        print('  %-14s %4d -> %4d cajas  (%s)' % (nom, len(filas) - nuevas, len(filas),
                                                  os.path.basename(sal)))
    print('\noriginales copiados a _para_borrar/gt_antes_de_auditoria_20092026')
    print('Los csv v2 NO se usan hasta que se apunte evaluate.py a ellos: medir con los dos.')


if __name__ == '__main__':
    main()

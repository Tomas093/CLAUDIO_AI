"""Reproduce tiles sinteticos de una muestra y dice de que sprite/generador salio cada caja (28/09).

Para la revision de Tomas (work/revision_tomas, grillas C-G): la muestra se armo con una version
anterior de compose.py (`--compose`) y VIS_MIN 0,85. Se regenera el tile con la misma semilla que usa
build_all.synth_job, se verifica que la imagen sea identica a la guardada y se devuelve, para cada caja,
el generador y (si vino de sym_instance) el nombre del archivo del sprite.

    py -3 src/origen_muestra.py work/ds21_muestra _para_borrar/compose_antes_vis90_28-09.py p000378 ...
"""
import os, sys, glob, random, importlib.util, json
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(__file__))


def cargar_compose(path):
    spec = importlib.util.spec_from_file_location('compose', path)
    m = importlib.util.module_from_spec(spec); sys.modules['compose'] = m; spec.loader.exec_module(m)
    return m


def nombres_lib(C):
    """Los nombres de archivo de LIB.syms, en el mismo orden que arma build_all."""
    from paths import DATA
    def lista(d, filtrar=True):
        out = []
        for f in sorted(glob.glob(os.path.join(d, '*.png'))):
            if os.path.basename(f) in C.EXCLUIR_SIM: continue
            g = cv2.imread(f, 0)
            if g is None: continue
            if filtrar and C.es_basura(g): continue
            out.append(os.path.relpath(f, DATA))
        return out
    full = os.path.join(DATA, 'sym_lib_full')
    n = lista(full) + lista(os.path.join(DATA, 'sym_lib'))[245:] * 3
    n += lista(os.path.join(DATA, 'sym_lib_tomas'), False) * int(os.environ.get('PESO_TOMAS', '1'))
    return n


def main():
    ds, comp, tiles = sys.argv[1], sys.argv[2], sys.argv[3:]
    from receta_ds21 import RECETA
    os.environ.update(RECETA); os.environ['VIS_MIN'] = os.environ.get('VIS_ORIGEN', '0.85')
    C = cargar_compose(comp)
    import build_all as B                     # usa el compose cargado arriba (sys.modules)
    nom = nombres_lib(C)
    assert len(nom) == len(B.LIB.syms), (len(nom), len(B.LIB.syms))
    reg = []
    orig_sym = C.sym_instance
    def espia(lib, r, txt_h):
        # sym_instance elige proc o sprite con el mismo r: se repite la eleccion con una copia del estado
        st = r.getstate(); proc = r.random() < .22
        k = r.randrange(len(C.GENS)) if proc else r.randrange(len(lib.syms))
        r.setstate(st)
        a = orig_sym(lib, r, txt_h)
        reg.append(('proc:' + C.GENS[k].__name__) if proc else nom[k])
        reg.append(a.shape if a is not None else None)
        return a
    C.sym_instance = espia
    gens = ['tablero_row', 'spm_branch', 'fusible_dxf', 'fila_densa', 'terna_rst', 'puls_contactor', 'marco_punteado',
            'trafo_o_pareja', 'fusible_solo', 'letra_recuadro', 'compuesto_vertical']
    quien = {}
    for g in gens:
        f = getattr(C, g)
        def w(*a, _f=f, _g=g, **k):
            boxes = a[4] if _g == 'letra_recuadro' else a[-1]
            n0 = len(boxes); res = _f(*a, **k)
            for b in boxes[n0:]: quien[id(b)] = _g
            return res
        setattr(C, g, w)
    out = {}
    for t in tiles:
        i = int(t[1:]); neg = t.startswith('n')
        r = random.Random((i * 7919 + (13 if neg else 0)) & 0xffffffff)
        reg.clear(); quien.clear()
        img, boxes = C.make_tile(B.LIB, r, negative=neg)
        ref = cv2.imread(os.path.join(ds, 'images', 'train', t + '.png'), 0)
        igual = ref is not None and ref.shape == img.shape and int(np.abs(ref.astype(int) - img.astype(int)).max()) == 0
        sprites = [(reg[j], reg[j + 1]) for j in range(0, len(reg), 2)]
        cajas = []
        for b in boxes:
            w, h = b[2] - b[0], b[3] - b[1]
            cand = [s for s, sh in sprites if sh is not None and (sh[1], sh[0]) == (w, h)]
            vis = 1.0
            if not cand:   # caja recortada por el borde: sprite con un lado igual y el otro mas largo
                cc = [(s, sh) for s, sh in sprites if sh is not None and
                      ((sh[1] == w and sh[0] > h) or (sh[0] == h and sh[1] > w))]
                if cc:
                    s, sh = max(cc, key=lambda z: w * h / (z[1][0] * z[1][1]))
                    cand = [s]; vis = round(w * h / (sh[0] * sh[1]), 3)
            cajas.append(dict(caja=[int(v) for v in b[:4]], generador=quien.get(id(b), 'sym'), sprite=cand[:3], vis=vis))
        out[t] = dict(identico=igual, cajas=cajas)
        print(t, 'identico' if igual else 'DISTINTO')
    json.dump(out, open(os.environ.get('ORIGEN_OUT', 'origen_muestra.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()

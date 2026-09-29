"""Convierte la biblioteca QET completa (solo categorias electricas de potencia/comando) a DXF limpio y la renderiza.
Salida: data/sym_lib_full/*.png (nombres anonimos) + reporte/grilla_biblioteca_completa_*.jpg para revisar."""
import sys, os, glob, re, cv2, numpy as np, ezdxf, zipfile
sys.path.insert(0, os.path.dirname(__file__))
from paths import BASE, DATA, WORK
from elmt2dxf import elmt_to_doc
from render import render_doc, plan_extents, CFG_COLOR
from symrender import tight
from numgrid import numbered_grid
from postproc import darken

ROOT = os.path.join(BASE, 'Base de simbolols', 'elementos_electricos')
INCLUDE = [r'10_allpole/110_network_supplies/.*(ground|terre|masse)', r'10_allpole/130_terminals', r'10_allpole/200_', r'10_allpole/310_',
           r'10_allpole/330_transformers_power_supplies/(10|40)', r'10_allpole/340_converters_inverters/(10|15|20)', r'10_allpole/380_signaling_operating/(11|12|20|21|25)',
           r'10_allpole/390_sensors_instruments/(60|70)', r'10_allpole/391_consumers_actuators/(10|50|60)', r'10_allpole/392_generators_sources/(10|20|30)',
           r'10_allpole/450_high_voltage/(?!.*(cle_|serrure|ligne|element_flexible|hvpole))', r'10_allpole/500_home_installation/40',
           r'11_singlepole/(200|330|392)', r'11_singlepole/500_home_installation/40', r'91_en_60617/en_60617_0[678]/']

EXCLUDE = [r'90_terminal_strips_diagram', r'10_generators/(pvcell|windmill)', r'20_power_units', r'en_60617_06_0[123]/',
           r'en_60617_07_0?1/', r'en_60617_07_1[12]/', r'en_60617_08_0[67]/']

def keep(rel, size=0):
    rel = rel.replace('\\', '/')
    if size > 15000: return False          # dibujos pictoricos / 3D / tablas enormes
    return any(re.search(p, rel) for p in INCLUDE) and not any(re.search(p, rel) for p in EXCLUDE)

if __name__ == '__main__':
    out = os.path.join(DATA, 'sym_lib_full'); os.makedirs(out, exist_ok=True)
    rep = os.path.join(WORK, 'reporte'); os.makedirs(rep, exist_ok=True)
    files = sorted(glob.glob(os.path.join(ROOT, '**', '*.elmt'), recursive=True))
    sel = [f for f in files if keep(os.path.relpath(f, ROOT), os.path.getsize(f))]
    print('[lib] elmt totales', len(files), 'seleccionados', len(sel))
    crops, k, rej = [], 0, 0
    for f in sel:
        try:
            doc, n = elmt_to_doc(f)
            if n < 2: rej += 1; continue
            x0, y0, x1, y1 = plan_extents(doc)
            if max(x1-x0, y1-y0) <= 0: rej += 1; continue
            img, _ = render_doc(doc, 160 / max(x1-x0, y1-y0), window=(x0, y0, x1, y1), pad_px=6, cfg=CFG_COLOR)
            t = tight(img)
            if t is None: rej += 1; continue
            h, w = t.shape
            ink = (darken(t, 3.5) < 128).mean()
            if max(h, w) / max(1, min(h, w)) > 8 or ink < .003 or ink > .6: rej += 1; continue   # lineas sueltas / manchas
            cv2.imwrite(os.path.join(out, f'e{k:05d}.png'), t); crops.append(darken(t, 3.5)); k += 1
        except Exception:
            rej += 1
    print('[lib] ok', k, 'rechazados', rej)
    for p in range(0, len(crops), 500):
        G = numbered_grid(crops[p:p+500], 'E', cell=110, cols=25, strip=16, start=p, digits=5,
                          title=f'BIBLIOTECA ELMT COMPLETA (positivos) E{p+1:05d}-E{min(len(crops), p+500):05d}')
        cv2.imwrite(os.path.join(rep, f'grilla_biblioteca_completa_{p//500+1:02d}.jpg'), G, [cv2.IMWRITE_JPEG_QUALITY, 85])

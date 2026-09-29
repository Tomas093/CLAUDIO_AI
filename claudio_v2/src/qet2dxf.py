"""Proyecto QElectroTech (.qet) -> un DXF por folio + cajas de elementos. Sin metadata (sin textos dinamicos de autor/uuid)."""
import sys, os, math, re, tempfile, xml.etree.ElementTree as ET, ezdxf
from ezdxf import bbox
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from elmt2dxf import elmt_to_doc
NO_COMP = re.compile(r'folio_referencing|sheet_referencing|previous_folio|next_folio|going_arrow|racourcie|renvoi|references|nomenclature|texte|teksty|linie_ramki|assembly_plan|thumbnails|98_graphics|pagegarde|vignettes|porte_etiquette|Grille_Accessoires|iso_sfc|grafcet|mounting|114_connections|odejscia_linii|/connections|pneum|oznaczenia|terminal_strips_diagram|network_settings|multifilaire/(pe|src_n|src_p|src_1pn|src_3p)|network_supplies/src|zrodl|/cable\\.elmt|^embed://import/cable|screen_big|/data\\.elmt|nouvel_element|TBX|temp_tb|en_tete|60_energy|arduino/gnd|pinball', re.I)

def defs(root):
    out = {}
    def walk(n, p):
        for ch in n:
            if ch.tag == 'category': walk(ch, p + '/' + ch.get('name'))
            elif ch.tag == 'element': out['embed://' + p.lstrip('/') + '/' + ch.get('name')] = ch.find('definition')
    col = root.find('collection')
    if col is not None: walk(col, '')
    return out

def rot(x, y, o):  # QET: orientacion 0..3 horaria, eje y hacia abajo
    for _ in range(int(o) % 4): x, y = -y, x
    return x, y

def folio_to_doc(root, d, D, cache):
    doc = ezdxf.new('R2010', setup=False); msp = doc.modelspace(); boxes = []; term = {}
    for e in d.findall('elements/element'):
        t = e.get('type'); ex, ey, o = float(e.get('x')), float(e.get('y')), int(e.get('orientation', 0))
        for tm in e.findall('terminals/terminal'):
            tx, ty = rot(float(tm.get('x')), float(tm.get('y')), o); term[tm.get('id')] = (ex + tx, -(ey + ty))
        if t not in D or D[t] is None: continue
        if t not in cache:
            f = tempfile.NamedTemporaryFile('wb', suffix='.elmt', delete=False); f.write(ET.tostring(D[t])); f.close()
            try: sub, n = elmt_to_doc(f.name)
            except Exception: sub, n = None, 0
            os.unlink(f.name)
            name = f'B{len(cache)}'
            if sub is not None and n:
                blk = doc.blocks.new(name)
                for ent in sub.modelspace():
                    if ent.dxftype() == 'HATCH' and ent.rgb == (255, 255, 255): continue   # relleno blanco: con politica BLACK saldria negro
                    try: blk.add_entity(ent.copy())
                    except Exception: pass
                cache[t] = name
            else: cache[t] = None
        if not cache[t]: continue
        ins = msp.add_blockref(cache[t], (ex, -ey), dxfattribs={'rotation': -90 * o})
        try:
            bb = bbox.extents(ins.virtual_entities())
            if bb.has_data and not NO_COMP.search(t):
                boxes.append([bb.extmin.x, bb.extmin.y, bb.extmax.x, bb.extmax.y, t])
        except Exception: pass
        lab = e.find("elementInformations/elementInformation[@name='label']")
        if lab is not None and lab.text: msp.add_text(lab.text[:12], height=7).set_placement((ex + 12, -ey + 10))
    for c in d.findall('conductors/conductor'):
        a, b = term.get(c.get('terminal1')), term.get(c.get('terminal2'))
        if a and b:
            mid = (a[0], b[1]) if abs(a[0]-b[0]) > 1 and abs(a[1]-b[1]) > 1 else b
            msp.add_lwpolyline([a, mid, b])
    W, H = float(d.get('cols', 17)) * float(d.get('colsize', 60)), float(d.get('rows', 8)) * float(d.get('rowsize', 80))
    msp.add_lwpolyline([(0, 0), (W, 0), (W, -H), (0, -H)], close=True)
    return doc, boxes

if __name__ == '__main__':
    src, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
    root = ET.parse(src).getroot(); D = defs(root); cache = {}
    for i, d in enumerate(root.findall('diagram')):
        doc, boxes = folio_to_doc(root, d, D, {})
        base = os.path.join(out, f'{os.path.basename(src)[:-4]}_f{i+1:02d}')
        doc.saveas(base + '.dxf')
        import json; json.dump(boxes, open(base + '.json', 'w'))
        print(base, len(boxes))

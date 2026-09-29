"""QET .elmt -> DXF limpio (sin metadata: sin nombres, uuid, autor, licencia, textos dinamicos)."""
import xml.etree.ElementTree as ET, ezdxf, math, re

def _f(e, k, d=0.0):
    try: return float(e.get(k, d))
    except: return d

def _fill(e):
    st = e.get('style', '')
    m = re.search(r'filling:([a-zA-Z#0-9]+)', st)
    f = m.group(1) if m else 'none'
    return None if f == 'none' else ('black' if f == 'black' else 'white')

def _hatch(msp, pts=None, circle=None, ellipse=None, color='white'):
    h = msp.add_hatch()
    h.rgb = (255, 255, 255) if color == 'white' else (0, 0, 0)
    if pts is not None: h.paths.add_polyline_path(pts, is_closed=True)
    elif circle is not None: h.paths.add_edge_path().add_arc(circle[0], circle[1], 0, 360)
    elif ellipse is not None: h.paths.add_edge_path().add_ellipse(ellipse[0], major_axis=ellipse[1], ratio=ellipse[2])

def elmt_to_doc(path, keep_static_text=True):
    root = ET.parse(path).getroot()
    desc = root.find('description')
    doc = ezdxf.new('R2010', setup=False)
    try:
        if 'EZDXF_META' in doc.rootdict: del doc.rootdict['EZDXF_META']
    except Exception: pass
    msp = doc.modelspace()
    n = 0
    for e in desc:
        t = e.tag
        if t == 'line':
            msp.add_line((_f(e,'x1'), -_f(e,'y1')), (_f(e,'x2'), -_f(e,'y2'))); n += 1
        elif t == 'polygon':
            pts = []; i = 1
            while e.get(f'x{i}') is not None:
                pts.append((_f(e,f'x{i}'), -_f(e,f'y{i}'))); i += 1
            closed = e.get('closed', 'true') != 'false'
            if len(pts) >= 2:
                fl = _fill(e)
                if fl and len(pts) >= 3: _hatch(msp, pts=pts, color=fl)
                msp.add_lwpolyline(pts, close=closed); n += 1
        elif t == 'rect':
            x, y, w, h = _f(e,'x'), _f(e,'y'), _f(e,'width'), _f(e,'height')
            pts = [(x,-y),(x+w,-y),(x+w,-y-h),(x,-y-h)]
            fl = _fill(e)
            if fl: _hatch(msp, pts=pts, color=fl)
            msp.add_lwpolyline(pts, close=True); n += 1
        elif t in ('ellipse', 'circle'):
            x, y = _f(e,'x'), _f(e,'y')
            w = _f(e,'width', _f(e,'diameter')); h = _f(e,'height', w)
            cx, cy = x + w/2, -(y + h/2)
            fl = _fill(e)
            if abs(w-h) < 1e-6:
                if fl: _hatch(msp, circle=((cx,cy), w/2), color=fl)
                msp.add_circle((cx,cy), w/2)
            elif w > 0 and h > 0:
                ma, ra = ((w/2,0), h/w) if w >= h else ((0,h/2), w/h)
                if fl: _hatch(msp, ellipse=((cx,cy), ma, ra), color=fl)
                msp.add_ellipse((cx,cy), major_axis=ma, ratio=ra)
            n += 1
        elif t == 'arc':
            x, y, w, h = _f(e,'x'), _f(e,'y'), _f(e,'width'), _f(e,'height')
            s, a = _f(e,'start'), _f(e,'angle')
            cx, cy = x + w/2, -(y + h/2)
            a0, a1 = (s, s+a) if a >= 0 else (s+a, s)
            if abs(w-h) < 1e-6:
                msp.add_arc((cx,cy), w/2, a0, a1)
            elif w > 0 and h > 0:
                if w >= h:
                    msp.add_ellipse((cx,cy), major_axis=(w/2,0), ratio=h/w, start_param=math.radians(a0), end_param=math.radians(a1))
                else:
                    msp.add_ellipse((cx,cy), major_axis=(0,h/2), ratio=w/h, start_param=math.radians(a0-90), end_param=math.radians(a1-90))
            n += 1
        elif t == 'text' and keep_static_text:
            txt = e.get('text', '')
            if txt and len(txt) <= 4:
                m = re.search(r',(\d+)', e.get('font', ',6'))
                size = float(m.group(1)) if m else 6
                msp.add_text(txt, height=size*0.9).set_placement((_f(e,'x'), -_f(e,'y')))
    return doc, n

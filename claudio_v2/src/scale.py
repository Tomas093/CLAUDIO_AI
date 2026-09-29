import numpy as np
TXT_PX = 11.0
def text_heights(doc):
    msp = doc.modelspace(); hs = []
    for t in msp.query('TEXT'): hs.append(t.dxf.height)
    for t in msp.query('MTEXT'): hs.append(t.dxf.char_height)
    for i in msp.query('INSERT'):
        try:
            for v in i.virtual_entities():
                if v.dxftype() == 'TEXT': hs.append(v.dxf.height)
                elif v.dxftype() == 'MTEXT': hs.append(v.dxf.char_height)
        except Exception: pass
    return np.array([h for h in hs if h and h > 0])
def auto_ppc(doc, txt_px=TXT_PX):
    hs = text_heights(doc)
    if len(hs) < 5: return None
    return txt_px / float(np.median(hs))

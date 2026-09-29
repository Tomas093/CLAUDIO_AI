import sys, numpy as np, cv2, ezdxf
sys.path.insert(0, __import__('os').path.dirname(__file__))
from render import render_doc, plan_extents

def render_symbol_doc(doc, target=140, entities=None, window=None):
    if window is None: window = plan_extents(doc)
    x0,y0,x1,y1 = window
    s = max(x1-x0, y1-y0, 1e-6)
    ppc = target / s
    img, meta = render_doc(doc, ppc, window=window, pad_px=6, entities=entities)
    return img

def tight(img, thr=200, pad=3):
    ys, xs = np.where(img < thr)
    if len(xs) == 0: return None
    y0, y1, x0, x1 = max(ys.min()-pad,0), min(ys.max()+pad+1,img.shape[0]), max(xs.min()-pad,0), min(xs.max()+pad+1,img.shape[1])
    return img[y0:y1, x0:x1]

def grid(crops, cell=110, cols=20, labels=None):
    rows = (len(crops)+cols-1)//cols
    G = np.full((rows*cell, cols*cell), 255, np.uint8)
    for i, c in enumerate(crops):
        h, w = c.shape[:2]; s = (cell-12)/max(h,w)
        c2 = cv2.resize(c, (max(1,int(w*s)), max(1,int(h*s))), interpolation=cv2.INTER_AREA)
        r, q = divmod(i, cols); y = r*cell + (cell-c2.shape[0])//2; x = q*cell + (cell-c2.shape[1])//2
        G[y:y+c2.shape[0], x:x+c2.shape[1]] = c2
        cv2.rectangle(G, (q*cell, r*cell), (q*cell+cell-1, r*cell+cell-1), 200, 1)
        if labels: cv2.putText(G, str(labels[i]), (q*cell+2, r*cell+10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 120, 1)
    return G

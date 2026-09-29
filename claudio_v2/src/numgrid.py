import numpy as np, cv2
def numbered_grid(crops, prefix, cell=150, cols=15, strip=20, title=None, colors=None, start=0, digits=4):
    rows = (len(crops) + cols - 1) // cols
    ch = cell + strip
    G = np.full((rows * ch, cols * cell, 3), 255, np.uint8)
    for i, c in enumerate(crops):
        if c.ndim == 2: c = cv2.cvtColor(c, cv2.COLOR_GRAY2BGR)
        h, w = c.shape[:2]; s = min((cell - 10) / max(h, w), 4.0)
        c2 = cv2.resize(c, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_NEAREST)
        r, q = divmod(i, cols); x0, y0 = q * cell, r * ch
        col = colors[i] if colors else (0, 0, 0)
        cv2.rectangle(G, (x0, y0), (x0 + cell - 1, y0 + strip), col, -1)
        cv2.putText(G, f'{prefix}{start+i+1:0{digits}d}', (x0 + 4, y0 + 15), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)
        yy = y0 + strip + (cell - c2.shape[0]) // 2; xx = x0 + (cell - c2.shape[1]) // 2
        G[yy:yy + c2.shape[0], xx:xx + c2.shape[1]] = c2
        cv2.rectangle(G, (x0, y0), (x0 + cell - 1, y0 + ch - 1), (170, 170, 170), 1)
    if title:
        hdr = np.full((50, G.shape[1], 3), 255, np.uint8)
        cv2.putText(hdr, title, (10, 34), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 0), 2, cv2.LINE_AA)
        G = np.vstack([hdr, G])
    return G

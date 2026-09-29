"""Regla detector + verificador (29/09). d = [x1,y1,x2,y2,conf,p].
Se descarta una deteccion SOLO si el verificador la rechaza (p < t_bajo) Y hay otra deteccion que la contiene
casi entera (IoS > ios) y que el verificador aprueba (p >= t_alto): es un pedazo o duplicado de un componente que
ya tiene su caja buena. Lo que el verificador no reconoce pero no compite con nada se deja (protege recall)."""
def aplicar(D, cmin=0.1, t_bajo=0.2, t_alto=0.5, ios=0.6, t_solo=None):
    D = [d for d in D if d[4] >= cmin]
    out = []
    for i, d in enumerate(D):
        if d[5] < t_bajo:
            A = max(1e-12, (d[2] - d[0]) * (d[3] - d[1])); tapado = False
            for j, k in enumerate(D):
                if j == i or k[5] < t_alto: continue
                ix = max(0, min(d[2], k[2]) - max(d[0], k[0])); iy = max(0, min(d[3], k[3]) - max(d[1], k[1]))
                B = max(1e-12, (k[2] - k[0]) * (k[3] - k[1]))
                if ix * iy / min(A, B) > ios: tapado = True; break
            if tapado: continue
            if t_solo is not None and d[5] < t_solo and d[4] < .5: continue   # opcional: sin caja buena al lado
        out.append(d)
    return out

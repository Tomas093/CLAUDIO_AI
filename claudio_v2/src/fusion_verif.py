"""Fusion guiada por el verificador (29/09): entre detecciones que se pisan (IoS > ios_max, sin mirar areas)
se queda la de mayor puntuacion del verificador; asi un pedazo con conf alta no le gana a la caja completa."""
def fusion(dets, ios_max=0.75, clave=lambda d: (d[5], d[4])):
    """dets: [x1,y1,x2,y2,conf,p]"""
    D = sorted(dets, key=clave, reverse=True); keep = []
    for d in D:
        A = max(1e-12, (d[2] - d[0]) * (d[3] - d[1])); ok = True
        for k in keep:
            ix = max(0, min(d[2], k[2]) - max(d[0], k[0])); iy = max(0, min(d[3], k[3]) - max(d[1], k[1]))
            B = max(1e-12, (k[2] - k[0]) * (k[3] - k[1]))
            if ix * iy / min(A, B) > ios_max: ok = False; break
        if ok: keep.append(d)
    return keep

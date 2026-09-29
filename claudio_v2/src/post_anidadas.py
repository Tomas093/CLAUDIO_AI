"""Postproceso extra (28/09, goal de Tomas: 100% recall con < 700 FP): descarta detecciones que quedan casi
enteras adentro de otra de MAYOR confianza (IoS > umbral), sin mirar la relacion de areas.

Motivo: de los 6.592 FP de RF4 a 0,05, 2.424 estan encima o adentro de un componente ya detectado (pedazos:
la cruz del interruptor, media lampara). `evaluate.fuse` solo suprime anidadas si las areas estan dentro de 4x
(para no comerse un TC dentro de una caja). En el GT de los 7 planos hay 25 pares anidados, todos
CONTACTOR+PULS con IoS 0,71, asi que el umbral se fija A PRIORI en 0,80 (no se ajusta mirando el test).
"""
def suprimir_anidadas(dets, ios_max=0.80):
    D = sorted(dets, key=lambda d: -d[4]); keep = []
    for d in D:
        A = max(1e-12, (d[2] - d[0]) * (d[3] - d[1])); ok = True
        for k in keep:
            ix = max(0, min(d[2], k[2]) - max(d[0], k[0])); iy = max(0, min(d[3], k[3]) - max(d[1], k[1]))
            B = max(1e-12, (k[2] - k[0]) * (k[3] - k[1]))
            if ix * iy / min(A, B) > ios_max: ok = False; break
        if ok: keep.append(d)
    return keep

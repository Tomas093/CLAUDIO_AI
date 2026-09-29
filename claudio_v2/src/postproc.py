import numpy as np
def darken(g, k=3.0):
    """Normaliza renders CAD: trazos finos antialiasados (gris claro) -> oscuros. Se usa igual en train e inferencia."""
    return (255 - np.clip((255.0 - g) * k, 0, 255)).astype(np.uint8)

def ink_map(bgr_or_gray):
    """0..1 tinta respecto del fondo (sirve para capturas a color con trazos claros)."""
    g = bgr_or_gray.min(2) if bgr_or_gray.ndim == 3 else bgr_or_gray
    g = g.astype(np.float32); bg = float(np.median(g))
    return np.clip((bg - g) / max(1.0, bg - 30.0), 0, 1)

def resize_keep_strokes(ink, s, gain=1.8):
    """Reescala conservando trazos finos: engrosa antes de achicar (si s<1) y devuelve gris 0-255 (tinta negra)."""
    import cv2
    if s < 1:
        k = max(1, int(round(1.0 / s)))
        if k > 1: ink = cv2.dilate(ink, np.ones((k, k), np.uint8))
        r = cv2.resize(ink, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    else:
        r = cv2.resize(ink, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
    return (255 - np.clip(r * gain, 0, 1) * 255).astype(np.uint8)

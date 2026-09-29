"""detectar.py - Detector de componentes en planos unifilares DXF (una clase: 'componente').

Script AUTOCONTENIDO para usar en el proyecto del usuario: solo necesita
ezdxf, matplotlib, opencv-python, numpy y ultralytics. No importa nada de claudio_v2/src.

Implementa la receta oficial de inferencia (identica a src/evaluate.py):
  1. Render del DXF CON TEXTO (color BLACK, fondo blanco) + darken.
  2. Escala automatica: la mediana de altura de texto queda en 11 px.
  3. Inferencia a 1.0x y 1.6x, tiles de 640 con stride 320, conf interna 0.05.
  4. Paso a coordenadas CAD y fusion NMS (IoU 0.45; anidadas con IoS 0.7 si las areas estan dentro de 4x).
  5. Recien ahi se aplica el umbral final que pide el usuario.

Uso:
    py -3 detectar.py <plano.dxf|carpeta> [--pesos best_limpio.pt] [--conf 0.25]
                      [--salida work/detecciones] [--escalas 1.0,1.6] [--device 0] [--sin-visual]

Salidas por plano en <salida>/:
    <plano>_detecciones.csv   x1,y1,x2,y2 en coordenadas CAD + conf
    <plano>_visual.png        render con las cajas dibujadas (omitible con --sin-visual)
    resumen.json              ppc usado, cantidad de detecciones y parametros
"""
import argparse, csv, glob, json, os
import cv2, numpy as np, ezdxf, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ezdxf.addons.drawing import RenderContext, Frontend
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend
from ezdxf.addons.drawing.config import Configuration, ColorPolicy, BackgroundPolicy, LineweightPolicy
from ezdxf import bbox as ebbox

TXT_PX = 11.0          # altura de texto objetivo en pixeles (define la escala del render)
TILE, STRIDE = 640, 320
CONF_INTERNA = 0.05    # se infiere bajo y se filtra al final: nunca perder un componente antes de fusionar

CFG = Configuration(color_policy=ColorPolicy.BLACK, background_policy=BackgroundPolicy.WHITE,
                    lineweight_policy=LineweightPolicy.ABSOLUTE, min_lineweight=0.25)

# ---------------------------------------------------------------- render / escala
def render_doc(doc, ppc, pad_px=32):
    e = ebbox.extents(doc.modelspace(), fast=True)
    x0, y0, x1, y1 = e.extmin.x, e.extmin.y, e.extmax.x, e.extmax.y
    pad = pad_px / ppc
    x0 -= pad; y0 -= pad; x1 += pad; y1 += pad
    W = int(round((x1 - x0) * ppc)); H = int(round((y1 - y0) * ppc))
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.axis("off")
    fig.patch.set_facecolor("white")
    Frontend(RenderContext(doc), MatplotlibBackend(ax), config=CFG).draw_layout(doc.modelspace(), finalize=False)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    plt.close(fig)
    g = cv2.cvtColor(buf, cv2.COLOR_RGB2GRAY)
    if g.shape != (H, W):
        g = cv2.resize(g, (W, H), interpolation=cv2.INTER_AREA)
    return g, dict(x0=x0, y1=y1, ppc=ppc, W=W, H=H)

def cad2px(m, x, y):
    return (x - m["x0"]) * m["ppc"], (m["y1"] - y) * m["ppc"]

def px2cad(m, px, py):
    return m["x0"] + px / m["ppc"], m["y1"] - py / m["ppc"]

def limpiar_mascaras(doc):
    """Saca lo que con politica BLACK se dibujaria como un bloque negro y taparia el plano.

    Tres casos vistos en planos reales:
      - WIPEOUT: mascara de AutoCAD, se pinta del color del fondo -> con BLACK sale negra.
      - HATCH solido blanco: relleno de tapado, mismo problema.
      - MTEXT con bg_fill: un bloque de notas con fondo relleno puede cubrir el plano entero
        (visto en hola.dxf: notas de 17.5x15.6 sobre un plano de 20x20). Se le apaga el fondo
        pero se conserva el texto, que hace falta para la escala automatica.
    Devuelve (entidades borradas, fondos de MTEXT apagados). Solo en memoria, no toca el DXF.
    """
    msp = doc.modelspace(); fuera = []; sin_fondo = 0
    for e in msp:
        t = e.dxftype()
        if t == 'WIPEOUT':
            fuera.append(e)
        elif t == 'HATCH' and getattr(e, 'solid_fill', False) and e.rgb == (255, 255, 255):
            fuera.append(e)
        elif t == 'MTEXT' and e.dxf.get('bg_fill', 0):
            e.dxf.bg_fill = 0
            sin_fondo += 1
    for e in fuera:
        msp.delete_entity(e)
    return len(fuera), sin_fondo

def auto_ppc(doc, txt_px=TXT_PX):
    """px por unidad CAD para que la mediana de altura de texto quede en txt_px."""
    msp = doc.modelspace(); hs = []
    for t in msp.query('TEXT'):
        hs.append(t.dxf.height)
    for t in msp.query('MTEXT'):
        hs.append(t.dxf.char_height)
    for i in msp.query('INSERT'):
        try:
            for v in i.virtual_entities():
                if v.dxftype() == 'TEXT':
                    hs.append(v.dxf.height)
                elif v.dxftype() == 'MTEXT':
                    hs.append(v.dxf.char_height)
        except Exception:
            pass
    hs = [h for h in hs if h and h > 0]
    if len(hs) < 5:
        return None          # plano sin texto suficiente: hay que pasar --ppc
    return txt_px / float(np.median(hs))

def darken(g, k=3.0):
    """Trazos finos antialiasados (gris claro) -> oscuros. Mismo preproceso que en entrenamiento."""
    return (255 - np.clip((255.0 - g) * k, 0, 255)).astype(np.uint8)

# ---------------------------------------------------------------- inferencia
def _suprimir(D, iou=.45, ios=.7, ar=4.0, fusionar=True, iou_fus=.55):
    """NMS + supresion de anidadas (solo si las areas estan dentro de 4x, para no comerse un TC dentro de una caja).

    Con `fusionar`, la caja que se descarta no se tira: se promedia contra la que queda,
    ponderando por confianza (box voting). Antes se conservaba siempre la de mas confianza,
    que suele ser la mas chica, y la caja terminaba cubriendo solo un pedazo del simbolo.
    Medido sobre EZE4077: area detectada/real 0.74 y IoU medio 0.466, cuando la escala 1.6
    sola daba 0.86 y 0.538. Solo se promedian cajas que claramente son el mismo objeto
    (IoU >= iou_fus); las anidadas chicas se descartan como antes.
    """
    D = sorted(D, key=lambda d: -d[4]); keep = []; peso = []
    for d in D:
        golpe = -1
        for n, k in enumerate(keep):
            ix = max(0, min(d[2], k[2]) - max(d[0], k[0]))
            iy = max(0, min(d[3], k[3]) - max(d[1], k[1]))
            I = ix * iy
            A = (d[2] - d[0]) * (d[3] - d[1]); B = (k[2] - k[0]) * (k[3] - k[1])
            j = I / (A + B - I + 1e-12)
            if j > iou or (I / (min(A, B) + 1e-12) > ios and max(A, B) < ar * min(A, B)):
                golpe = n if j >= iou_fus else -2      # -2 = descartar sin fusionar
                break
        if golpe == -1:
            keep.append(list(d)); peso.append(d[4])
        elif golpe >= 0 and fusionar:
            k = keep[golpe]; w0 = peso[golpe]; w1 = d[4]; w = w0 + w1
            for t in range(4):
                k[t] = (k[t] * w0 + d[t] * w1) / w
            k[4] = max(k[4], d[4])
            peso[golpe] = w
    return [tuple(k) for k in keep]

def infer_tiles(model, img, conf=CONF_INTERNA, tile=TILE, stride=STRIDE, device=None, batch_n=32):
    H, W = img.shape; dets = []
    ys = list(range(0, max(1, H - tile) + 1, stride)); xs = list(range(0, max(1, W - tile) + 1, stride))
    if ys[-1] + tile < H: ys.append(H - tile)
    if xs[-1] + tile < W: xs.append(W - tile)
    batch, offs = [], []

    def run():
        kw = dict(imgsz=tile, conf=conf, verbose=False)
        if device is not None:
            kw['device'] = device
        for r, (ox, oy) in zip(model.predict(batch, **kw), offs):
            for b, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
                dets.append([b[0] + ox, b[1] + oy, b[2] + ox, b[3] + oy, c])
        batch.clear(); offs.clear()

    for y in ys:
        for x in xs:
            t = img[max(0, y):y + tile, max(0, x):x + tile]
            if t.shape != (tile, tile):
                t = cv2.copyMakeBorder(t, 0, tile - t.shape[0], 0, tile - t.shape[1], cv2.BORDER_CONSTANT, value=255)
            if (t < 128).sum() == 0:
                continue        # tile sin tinta: no aporta nada
            batch.append(cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)); offs.append((max(0, x), max(0, y)))
            if len(batch) == batch_n:
                run()
    if batch:
        run()
    return _suprimir(dets)

def render_pdf(pdf_path, pagina=0, txt_px=TXT_PX, dpi=None):
    """Renderiza una pagina de PDF vectorial a gris, a la escala que usa el detector.

    Se va derecho del PDF a la imagen, sin pasar por DXF: PyMuPDF ya respeta la rotacion de
    pagina, las fuentes y los anchos de linea, y asi no se arrastran los errores de una
    conversion vectorial. La escala se elige igual que en DXF: que la mediana de altura de
    texto quede en `txt_px`. Devuelve (img_gris, meta) con meta para volver a puntos PDF.
    """
    try:
        import fitz
    except ImportError:
        raise SystemExit('para leer PDF hace falta PyMuPDF: py -3 -m pip install pymupdf')
    doc = fitz.open(pdf_path)
    page = doc[pagina]
    if dpi is None:
        alturas = []
        for blk in page.get_text('dict').get('blocks', []):
            for line in blk.get('lines', []):
                for sp in line.get('spans', []):
                    if sp['text'].strip() and sp['size'] > 0:
                        alturas.append(sp['size'])
        if len(alturas) < 5:
            doc.close()
            raise SystemExit('%s: el texto del PDF esta vectorizado (%d textos), no se puede '
                             'autoescalar. Pasar --dpi.' % (os.path.basename(pdf_path), len(alturas)))
        alturas.sort()
        mediana = alturas[len(alturas) // 2]
        dpi = txt_px * 72.0 / mediana        # 1 punto = dpi/72 px
    dpi = int(round(dpi))                    # get_pixmap solo acepta dpi entero
    pm = page.get_pixmap(dpi=dpi)
    img = np.frombuffer(pm.samples, dtype=np.uint8).reshape(pm.height, pm.width, pm.n)
    if pm.n >= 3:
        img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2GRAY)
    else:
        img = img[:, :, 0]
    doc.close()
    esc = dpi / 72.0                         # px por punto PDF
    return img, dict(esc=esc, dpi=dpi, W=pm.width, H=pm.height)

def px2pdf(meta, px, py):
    """Pixel del render -> punto de la pagina PDF (y hacia abajo, como en el PDF)."""
    return px / meta['esc'], py / meta['esc']

def detectar_pdf(model, pdf_path, escalas=(1.0, 1.6), conf=0.25, dpi=None, device=None,
                 visual=True, batch_n=32, pagina=0):
    """Igual que detectar_dxf pero sobre un PDF. Coordenadas de salida en puntos PDF."""
    img0, meta0 = render_pdf(pdf_path, pagina, dpi=dpi)
    img0 = darken(img0)
    dets = []
    for sc in escalas:
        img = img0 if sc == 1.0 else cv2.resize(img0, None, fx=sc, fy=sc, interpolation=cv2.INTER_LINEAR)
        for d in infer_tiles(model, img, device=device, batch_n=batch_n):
            dets.append([d[0] / sc, d[1] / sc, d[2] / sc, d[3] / sc, d[4]])   # todo a pixeles de 1.0x
    if len(escalas) > 1:
        dets = _suprimir(dets)
    dets = [d for d in dets if d[4] >= conf]
    vis = None
    if visual:
        vis = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)
        for d in dets:
            cv2.rectangle(vis, (int(d[0]), int(d[1])), (int(d[2]), int(d[3])), (0, 170, 0), 2)
            cv2.putText(vis, '%.2f' % d[4], (int(d[0]), int(d[1]) - 3), 0, .4, (0, 130, 0), 1)
    pdfc = [[*px2pdf(meta0, d[0], d[1]), *px2pdf(meta0, d[2], d[3]), d[4]] for d in dets]
    return pdfc, vis, meta0

def detectar_dxf(model, dxf_path, escalas=(1.0, 1.6), conf=0.25, ppc=None, device=None, visual=True, batch_n=32):
    """Devuelve (lista de [x1,y1,x2,y2,conf] en CAD ya filtrada, imagen visual o None, ppc)."""
    doc = ezdxf.readfile(dxf_path)
    n_masc, n_fondo = limpiar_mascaras(doc)
    if n_masc or n_fondo:
        print('  (se ignoraron %d mascaras/rellenos y %d fondos de MTEXT que taparian el plano)' % (n_masc, n_fondo))
    if ppc is None:
        ppc = auto_ppc(doc)
        if ppc is None:
            raise SystemExit("%s: menos de 5 textos, no se puede autoescalar. Pasar --ppc." % os.path.basename(dxf_path))
    dc, img0, meta0 = [], None, None
    for k, sc in enumerate(escalas):
        img, meta = render_doc(doc, ppc * sc)
        img = darken(img)
        for d in infer_tiles(model, img, device=device, batch_n=batch_n):
            xa, ya = px2cad(meta, d[0], d[3]); xb, yb = px2cad(meta, d[2], d[1])
            dc.append([xa, ya, xb, yb, d[4]])
        if k == 0:
            img0, meta0 = img, meta
    if len(escalas) > 1:
        dc = _suprimir(dc)          # fusion entre escalas, en coordenadas CAD
    dc = [d for d in dc if d[4] >= conf]             # umbral final, recien al final
    vis = None
    if visual:
        vis = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)
        for d in dc:
            x1, y1 = cad2px(meta0, d[0], d[3]); x2, y2 = cad2px(meta0, d[2], d[1])
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 170, 0), 2)
            cv2.putText(vis, '%.2f' % d[4], (int(x1), int(y1) - 3), 0, .4, (0, 130, 0), 1)
    return dc, vis, ppc

# ---------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description="Detector de componentes en planos unifilares DXF")
    ap.add_argument('entrada', help='archivo .dxf o carpeta con .dxf')
    ap.add_argument('--pesos', default='best_limpio.pt')
    ap.add_argument('--conf', type=float, default=0.25, help='umbral final (0.25 default; bajar a 0.10/0.05 prioriza recall)')
    ap.add_argument('--salida', default=os.path.join('work', 'detecciones'))
    ap.add_argument('--escalas', default='1.0,1.6')
    ap.add_argument('--ppc', type=float, default=None, help='forzar px por unidad CAD (planos sin texto)')
    ap.add_argument('--device', default=None, help="'0' para GPU, 'cpu' para CPU")
    ap.add_argument('--batch', type=int, default=32, help='tiles por lote (bajar si la GPU esta ocupada)')
    ap.add_argument('--dpi', type=float, default=None, help='solo PDF: forzar dpi de render (si el texto esta vectorizado)')
    ap.add_argument('--sin-visual', action='store_true')
    a = ap.parse_args()

    from ultralytics import YOLO
    if not os.path.exists(a.pesos):
        raise SystemExit("no existe el archivo de pesos: %s" % a.pesos)
    model = YOLO(a.pesos)
    escalas = tuple(float(s) for s in a.escalas.split(','))
    if os.path.isdir(a.entrada):
        planos = sorted(glob.glob(os.path.join(a.entrada, '*.dxf')) + glob.glob(os.path.join(a.entrada, '*.pdf')))
    else:
        planos = [a.entrada]
    if not planos:
        raise SystemExit("no se encontraron .dxf ni .pdf en %s" % a.entrada)
    os.makedirs(a.salida, exist_ok=True)
    resumen = dict(pesos=os.path.abspath(a.pesos), conf=a.conf, escalas=list(escalas), tile=TILE, stride=STRIDE, planos={})
    for p in planos:
        name = os.path.splitext(os.path.basename(p))[0]
        try:
            if p.lower().endswith('.pdf'):
                # El PDF se renderiza directo a imagen: no se pasa por DXF, asi no se arrastran
                # errores de conversion vectorial. Las coordenadas salen en puntos de la pagina.
                dets, vis, meta = detectar_pdf(model, p, escalas, a.conf, a.dpi, a.device, not a.sin_visual, a.batch)
                unidad, medida = 'dpi', meta['dpi']
            else:
                dets, vis, medida = detectar_dxf(model, p, escalas, a.conf, a.ppc, a.device, not a.sin_visual, a.batch)
                unidad = 'ppc'
        except SystemExit as e:      # un plano sin escala no corta el lote
            print('%s: SALTEADO -> %s' % (name, e))
            resumen['planos'][name] = dict(error=str(e))
            continue
        with open(os.path.join(a.salida, name + '_detecciones.csv'), 'w', newline='', encoding='utf-8') as f:
            wr = csv.writer(f); wr.writerow(['x1', 'y1', 'x2', 'y2', 'conf'])
            for d in dets:
                wr.writerow([round(v, 4) for v in d[:4]] + [round(d[4], 3)])
        if vis is not None:
            cv2.imwrite(os.path.join(a.salida, name + '_visual.png'), vis)
        resumen['planos'][name] = {unidad: round(medida, 3), 'detecciones': len(dets)}
        print('%s: %d componentes (%s %.2f)' % (name, len(dets), unidad, medida))
    json.dump(resumen, open(os.path.join(a.salida, 'resumen.json'), 'w'), indent=1, ensure_ascii=False)
    print('salida en', os.path.abspath(a.salida))

if __name__ == '__main__':
    main()

import ezdxf, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ezdxf.addons.drawing import RenderContext, Frontend
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend
from ezdxf.addons.drawing.config import Configuration, ColorPolicy, BackgroundPolicy, LineweightPolicy
from ezdxf import bbox as ebbox
import cv2

CFG = Configuration(color_policy=ColorPolicy.BLACK, background_policy=BackgroundPolicy.WHITE,
                    lineweight_policy=LineweightPolicy.ABSOLUTE, min_lineweight=0.25)

def plan_extents(doc):
    e = ebbox.extents(doc.modelspace(), fast=True)
    return e.extmin.x, e.extmin.y, e.extmax.x, e.extmax.y

CFG_COLOR = Configuration(color_policy=ColorPolicy.COLOR, background_policy=BackgroundPolicy.WHITE, lineweight_policy=LineweightPolicy.ABSOLUTE, min_lineweight=0.25)

def render_doc(doc, ppc, window=None, pad_px=32, entities=None, cfg=None):
    """Render modelspace (or entity list) to grayscale uint8. Returns img, meta (x0,y1 top-left CAD, ppc)."""
    if window is None:
        window = plan_extents(doc)
    x0, y0, x1, y1 = window
    pad = pad_px / ppc
    x0 -= pad; y0 -= pad; x1 += pad; y1 += pad
    W = int(round((x1 - x0) * ppc)); H = int(round((y1 - y0) * ppc))
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.axis("off")
    fig.patch.set_facecolor("white")
    ctx = RenderContext(doc)
    fe = Frontend(ctx, MatplotlibBackend(ax), config=cfg or CFG)
    if entities is None:
        fe.draw_layout(doc.modelspace(), finalize=False)
    else:
        fe.draw_entities(entities)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    plt.close(fig)
    g = cv2.cvtColor(buf, cv2.COLOR_RGB2GRAY)
    if g.shape != (H, W):
        g = cv2.resize(g, (W, H), interpolation=cv2.INTER_AREA)
    return g, dict(x0=x0, y1=y1, ppc=ppc, W=W, H=H)

def cad2px(meta, x, y):
    return (x - meta["x0"]) * meta["ppc"], (meta["y1"] - y) * meta["ppc"]

def px2cad(meta, px, py):
    return meta["x0"] + px / meta["ppc"], meta["y1"] - py / meta["ppc"]

# generate_backgrounds.py — Fondos reales a partir de planos DXF completos.
#
# Cambios respecto de la versión anterior (ver auditoría H3):
#   * El directorio de salida se limpia antes de generar (antes se acumulaba
#     basura de corridas viejas, incluidos los 500 white_*.jpg).
#   * Todas las entidades se fuerzan a negro, igual que en phase1_extractor,
#     para no depender del color de fondo que devuelva el backend de ezdxf.
#   * El render es determinista: se fija px_per_unit a partir del tamaño real
#     de los símbolos del plano, así el fondo sale a la MISMA escala a la que
#     después se va a hacer inferencia.
#   * Planos grandes se renderizan por ventanas, no de una (background.dxf
#     entero a 54 px/unidad son ~300 M de píxeles).
#   * BORRADO DE SÍMBOLOS: se eliminan del render los INSERT, los círculos y
#     hatches de tamaño de símbolo, y las entidades contenidas dentro de
#     ellos. Sin esto, usar planos reales como fondo reintroduce exactamente
#     el bug H1 — instancias del target sin etiquetar dentro de la imagen.
#     Se borran en vez de descartar el tile entero porque descartar deja
#     solamente los márgenes en blanco del plano.
from __future__ import annotations

import io
import math
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import cv2
import ezdxf
import matplotlib
import numpy as np
from ezdxf import bbox as ezdxf_bbox
from ezdxf.addons.drawing import Frontend, RenderContext
from ezdxf.addons.drawing.config import Configuration
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import load_config


# ── Geometría de rechazo ──────────────────────────────────────────────────────

@dataclass
class Rect:
    """Rectángulo en unidades CAD."""
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def max_side(self) -> float:
        return max(self.x1 - self.x0, self.y1 - self.y0)

    def expanded(self, m: float) -> "Rect":
        return Rect(self.x0 - m, self.y0 - m, self.x1 + m, self.y1 + m)


def _entity_rect(entity) -> Rect | None:
    try:
        bb = ezdxf_bbox.extents([entity])
    except Exception:
        return None
    if not bb.has_data:
        return None
    return Rect(bb.extmin.x, bb.extmin.y, bb.extmax.x, bb.extmax.y)


def _block_local_rect(doc, block_name: str, _cache: dict = {}) -> Rect | None:
    """BBox de la definición de un bloque, cacheado por nombre.

    Calcular ``ezdxf.bbox.extents`` por cada INSERT expande el bloque una vez
    por inserción: con 1199 inserciones tarda minutos. Se calcula una vez por
    definición y después se le aplica la transformación de cada inserción.
    """
    key = (id(doc), block_name)
    if key in _cache:
        return _cache[key]
    rect = None
    try:
        block = doc.blocks.get(block_name)
        if block is not None:
            bb = ezdxf_bbox.extents(block, fast=True)
            if bb.has_data:
                rect = Rect(bb.extmin.x, bb.extmin.y, bb.extmax.x, bb.extmax.y)
    except Exception:
        rect = None
    _cache[key] = rect
    return rect


def _insert_rect(doc, insert) -> Rect | None:
    """BBox en coordenadas de modelspace de una inserción de bloque."""
    local = _block_local_rect(doc, insert.dxf.name)
    if local is None:
        return _entity_rect(insert)
    try:
        sx = float(getattr(insert.dxf, "xscale", 1.0) or 1.0)
        sy = float(getattr(insert.dxf, "yscale", 1.0) or 1.0)
        rot = math.radians(float(getattr(insert.dxf, "rotation", 0.0) or 0.0))
        ox, oy = float(insert.dxf.insert.x), float(insert.dxf.insert.y)
    except Exception:
        return _entity_rect(insert)

    cos_r, sin_r = math.cos(rot), math.sin(rot)
    xs, ys = [], []
    for cx, cy in (
        (local.x0, local.y0), (local.x1, local.y0),
        (local.x1, local.y1), (local.x0, local.y1),
    ):
        px, py = cx * sx, cy * sy
        xs.append(ox + px * cos_r - py * sin_r)
        ys.append(oy + px * sin_r + py * cos_r)
    return Rect(min(xs), min(ys), max(xs), max(ys))


def collect_symbol_rects(
    msp,
    min_circle_radius: float,
    min_hatch_side: float,
    doc=None,
) -> tuple[list[Rect], set[int], dict[str, int]]:
    """Devuelve los rectángulos CAD ocupados por algo que puede ser un símbolo.

    Tres fuentes, porque en estos planos los símbolos aparecen de las tres
    formas:

      1. ``INSERT`` — el caso limpio: el símbolo es un bloque insertado.
      2. ``CIRCLE`` de radio significativo — muchos símbolos (ojo_de_buey,
         circulo_M, grupo_electrogeno, instrumento de medición) son un
         círculo dibujado como geometría suelta, sin bloque. Los círculos
         chicos son puntos de conexión de cables y no se rechazan.
      3. ``HATCH`` de tamaño significativo — rellenos que suelen ser parte
         de un símbolo.

    Lo que NO se rechaza es justamente lo que queremos como fondo: líneas de
    conexión, polilíneas, textos, cotas, marcos y tablas.
    """
    rects: list[Rect] = []
    ent_ids: set[int] = set()
    stats = {"insert": 0, "circle": 0, "hatch": 0, "skipped_small": 0}

    doc = doc if doc is not None else msp.doc
    insert_sides: list[float] = []
    for insert in msp.query("INSERT"):
        r = _insert_rect(doc, insert)
        if r is not None:
            rects.append(r)
            ent_ids.add(id(insert))
            insert_sides.append(r.max_side)
            stats["insert"] += 1
    stats["insert_sides"] = insert_sides

    for circle in msp.query("CIRCLE"):
        try:
            radius = float(circle.dxf.radius)
        except Exception:
            continue
        if radius < min_circle_radius:
            stats["skipped_small"] += 1
            continue
        cx, cy = float(circle.dxf.center.x), float(circle.dxf.center.y)
        rects.append(Rect(cx - radius, cy - radius, cx + radius, cy + radius))
        ent_ids.add(id(circle))
        stats["circle"] += 1

    for hatch in msp.query("HATCH"):
        r = _entity_rect(hatch)
        if r is None:
            continue
        if r.max_side < min_hatch_side:
            stats["skipped_small"] += 1
            continue
        rects.append(r)
        ent_ids.add(id(hatch))
        stats["hatch"] += 1

    return rects, ent_ids, stats


# ── Render determinista ───────────────────────────────────────────────────────

def _force_black(doc) -> None:
    """Fuerza todas las entidades a negro puro (mismo criterio que phase1)."""
    for layer in doc.layers:
        try:
            layer.color = 250
        except Exception:
            pass

    def _black(entity):
        try:
            entity.dxf.true_color = 0x000000
        except Exception:
            pass

    for entity in doc.modelspace():
        _black(entity)
    for block in doc.blocks:
        for entity in block:
            _black(entity)


class RectGrid:
    """Índice espacial grueso para consultar "¿qué símbolos tocan este rect?".

    Sin esto, comprobar 19 000 entidades contra 1 800 rectángulos de símbolo
    son 34 millones de comparaciones por plano.
    """

    def __init__(self, rects: list[Rect], cell: float):
        self.cell = max(cell, 1e-6)
        self.buckets: dict[tuple[int, int], list[Rect]] = {}
        for r in rects:
            for key in self._keys(r):
                self.buckets.setdefault(key, []).append(r)

    def _keys(self, r: Rect):
        for gx in range(int(r.x0 // self.cell), int(r.x1 // self.cell) + 1):
            for gy in range(int(r.y0 // self.cell), int(r.y1 // self.cell) + 1):
                yield (gx, gy)

    def contained_in_any(self, r: Rect) -> bool:
        """True si *r* está completamente dentro de algún rectángulo del índice."""
        seen: set[int] = set()
        for key in self._keys(r):
            for other in self.buckets.get(key, ()):
                if id(other) in seen:
                    continue
                seen.add(id(other))
                if (other.x0 <= r.x0 and other.y0 <= r.y0
                        and other.x1 >= r.x1 and other.y1 >= r.y1):
                    return True
        return False


def select_background_entities(
    indexed: list[tuple[Rect, object]],
    symbol_entities: set[int],
    symbol_rects: list[Rect],
    cell: float,
) -> tuple[list[tuple[Rect, object]], dict[str, int]]:
    """Quita del plano todo lo que sea un símbolo, y deja el resto.

    Rechazar los tiles que contienen símbolos deja solamente los márgenes del
    plano: tiles casi en blanco, que es justo lo que queríamos evitar. En
    cambio, si se BORRAN las entidades del símbolo y se renderiza el resto, el
    tile conserva toda la densidad real del plano —cables, textos, cotas,
    marcos, tablas— sin ninguna instancia del target sin etiquetar.

    Se descartan dos cosas:
      * las entidades identificadas como símbolo (INSERT, círculos grandes,
        hatches grandes);
      * cualquier entidad contenida por completo dentro del rectángulo de un
        símbolo, que son las líneas internas del propio símbolo dibujadas
        como geometría suelta.
    """
    grid = RectGrid(symbol_rects, cell)
    kept: list[tuple[Rect, object]] = []
    stats = {"symbol": 0, "inside_symbol": 0, "kept": 0}
    for rect, ent in indexed:
        if id(ent) in symbol_entities:
            stats["symbol"] += 1
            continue
        if grid.contained_in_any(rect):
            stats["inside_symbol"] += 1
            continue
        kept.append((rect, ent))
    stats["kept"] = len(kept)
    return kept, stats


def index_entities(msp) -> list[tuple[Rect, object]]:
    """Precalcula el bbox de cada entidad del modelspace, una sola vez.

    Sin esto cada ventana redibuja las 19 000 entidades del plano completo y
    tarda ~2 minutos; con el índice, cada ventana dibuja solo lo que cae
    adentro y tarda menos de un segundo.
    """
    entities = list(msp)
    cache = ezdxf_bbox.Cache()
    indexed: list[tuple[Rect, object]] = []
    for ent, box in zip(entities, ezdxf_bbox.multi_flat(entities, cache=cache, fast=True)):
        if box is None or not box.has_data:
            continue
        indexed.append(
            (Rect(box.extmin.x, box.extmin.y, box.extmax.x, box.extmax.y), ent)
        )
    return indexed


def render_window(doc, indexed: list[tuple[Rect, object]],
                  x0: float, y0: float, x1: float, y1: float,
                  width_px: int, height_px: int) -> np.ndarray | None:
    """Renderiza la ventana CAD [x0,x1]×[y0,y1] a una imagen de width_px×height_px.

    El eje ocupa toda la figura, así que el mapeo CAD→píxel es lineal y exacto:
    ``px = (x - x0) / (x1 - x0) * W``  y  ``py = (1 - (y - y0)/(y1 - y0)) * H``.
    """
    visible = [e for (r, e) in indexed
               if r.x1 >= x0 and r.x0 <= x1 and r.y1 >= y0 and r.y0 <= y1]
    if not visible:
        return None

    dpi = 100
    fig = plt.figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
    try:
        fig.patch.set_facecolor("white")
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_facecolor("white")
        ax.axis("off")

        ctx = RenderContext(doc)
        backend = MatplotlibBackend(ax)
        frontend = Frontend(ctx, backend, config=Configuration.defaults())
        frontend.draw_entities(visible)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, facecolor="white", edgecolor="none")
        buf.seek(0)
        img = cv2.imdecode(np.frombuffer(buf.read(), np.uint8), cv2.IMREAD_COLOR)
        buf.close()
        return img
    except Exception as exc:  # pragma: no cover - depende del DXF
        print(f"    ! Error renderizando ventana: {exc}")
        return None
    finally:
        plt.close(fig)


# ── Tiling ────────────────────────────────────────────────────────────────────

def _rect_to_px(r: Rect, x0: float, y0: float, x1: float, y1: float,
                W: int, H: int) -> tuple[int, int, int, int]:
    sx = W / (x1 - x0)
    sy = H / (y1 - y0)
    px0 = (r.x0 - x0) * sx
    px1 = (r.x1 - x0) * sx
    # el eje Y del CAD va hacia arriba y el de la imagen hacia abajo
    py0 = (1.0 - (r.y1 - y0) / (y1 - y0)) * H
    py1 = (1.0 - (r.y0 - y0) / (y1 - y0)) * H
    return int(math.floor(px0)), int(math.floor(py0)), int(math.ceil(px1)), int(math.ceil(py1))


def _overlaps_any(tile: tuple[int, int, int, int],
                  rects_px: list[tuple[int, int, int, int]]) -> bool:
    tx0, ty0, tx1, ty1 = tile
    for rx0, ry0, rx1, ry1 in rects_px:
        if tx0 < rx1 and tx1 > rx0 and ty0 < ry1 and ty1 > ry0:
            return True
    return False


# ── Entrada principal ─────────────────────────────────────────────────────────

def generate_backgrounds(
    dxf_sources_dir: Path | None = None,
    output_dir: Path | None = None,
    tile_size: int | None = None,
    overlap: int | None = None,
    min_std_dev: float | None = None,
    target_symbol_px: int | None = None,
    max_tiles_per_plan: int | None = None,
    seed: int = 42,
    wipe: bool = True,
) -> int:
    """Genera tiles de fondo desde todos los DXF de *dxf_sources_dir*.

    Devuelve la cantidad total de tiles guardados.
    """
    cfg = load_config()
    bg_cfg = cfg.backgrounds

    dxf_sources_dir = dxf_sources_dir or bg_cfg.dxf_sources_dir
    output_dir = output_dir or cfg.g.backgrounds_dir
    tile_size = tile_size if tile_size is not None else bg_cfg.tile_size
    overlap = overlap if overlap is not None else bg_cfg.overlap
    min_std_dev = min_std_dev if min_std_dev is not None else bg_cfg.min_std_dev
    target_symbol_px = (
        target_symbol_px if target_symbol_px is not None else bg_cfg.target_symbol_px
    )
    max_tiles_per_plan = (
        max_tiles_per_plan if max_tiles_per_plan is not None else bg_cfg.max_tiles_per_plan
    )

    rng = random.Random(seed)

    if not dxf_sources_dir.exists():
        dxf_sources_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"[Fondos] Creada carpeta '{dxf_sources_dir}'.\n"
            f"  Poné ahí los planos DXF completos y volvé a ejecutar."
        )
        return 0

    planos = sorted(
        p for p in dxf_sources_dir.glob("*.dxf")
        if p.stem.lower() not in bg_cfg.exclude_stems
    )
    if not planos:
        print(f"[Fondos] No se encontraron .dxf utilizables en '{dxf_sources_dir}'")
        return 0

    # Idempotencia: sin esto los fondos viejos (los white_*.jpg) sobreviven
    # y se mezclan con los nuevos.
    if wipe and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    step = tile_size - overlap
    if step <= 0:
        raise ValueError(f"overlap ({overlap}) debe ser menor que tile_size ({tile_size})")

    print(f"[Fondos] {len(planos)} planos | tile {tile_size}px | paso {step}px | "
          f"símbolo objetivo ~{target_symbol_px}px")

    total = 0
    for plano in planos:
        n = _process_plan(
            plano, output_dir, tile_size, step, min_std_dev,
            target_symbol_px, max_tiles_per_plan, bg_cfg, rng,
        )
        total += n

    print(f"\n[Fondos] ✅ {total} tiles guardados en '{output_dir}'")
    if total == 0:
        raise RuntimeError(
            "No se generó ningún fondo. Revisá los planos de entrada o bajá "
            "min_std_dev / min_circle_radius en el bloque `backgrounds:` del YAML."
        )
    return total


def _process_plan(plano: Path, output_dir: Path, tile_size: int, step: int,
                  min_std_dev: float, target_symbol_px: int,
                  max_tiles_per_plan: int, bg_cfg, rng: random.Random) -> int:
    print(f"\n  {plano.name}")
    try:
        doc = ezdxf.readfile(str(plano))
    except Exception as exc:
        print(f"    ! No se pudo leer: {exc}")
        return 0

    msp = doc.modelspace()
    _force_black(doc)

    try:
        bb = ezdxf_bbox.extents(msp)
    except Exception as exc:
        print(f"    ! Sin bbox: {exc}")
        return 0
    if not bb.has_data:
        print("    ! ModelSpace vacío")
        return 0

    X0, Y0 = bb.extmin.x, bb.extmin.y
    X1, Y1 = bb.extmax.x, bb.extmax.y

    sym_rects, sym_ids, stats = collect_symbol_rects(
        msp,
        min_circle_radius=bg_cfg.min_circle_radius,
        min_hatch_side=bg_cfg.min_hatch_side,
        doc=doc,
    )
    indexed = index_entities(msp)

    # Escala: los símbolos de estos planos miden ~1.19 unidades CAD de lado.
    # Fijamos px_per_unit para que rindan a target_symbol_px, la misma escala
    # a la que dxf_to_image renderiza para inferencia.
    #
    # Se mide SOLO sobre INSERT y descartando los bloques diminutos (flechas,
    # marcas de cota, viñetas): tomar la mediana de todo mezclaba símbolos
    # reales con marcadores y daba escalas 3× distintas entre planos, lo que
    # rompe la consistencia del contexto entre fondos.
    ref = bg_cfg.fallback_symbol_cad
    sides = np.array([s for s in stats.get("insert_sides", []) if s >= ref * 0.25])
    symbol_cad = float(np.median(sides)) if sides.size else ref
    if not (ref * 0.4 <= symbol_cad <= ref * 2.5):
        print(f"    ! símbolo medido {symbol_cad:.3f} CAD fuera de rango; "
              f"se usa la referencia {ref:.3f}")
        symbol_cad = ref
    px_per_unit = target_symbol_px / symbol_cad

    margin_cad = bg_cfg.symbol_margin_px / px_per_unit
    sym_rects = [r.expanded(margin_cad) for r in sym_rects]

    # Se BORRAN los símbolos del plano en vez de descartar los tiles que los
    # contienen: así el fondo conserva la densidad real (cables, textos,
    # cotas, marcos, tablas) sin ninguna instancia del target sin etiquetar.
    indexed, sel_stats = select_background_entities(
        indexed, sym_ids, sym_rects, cell=max(symbol_cad * 4, 1e-3)
    )

    total_w = int((X1 - X0) * px_per_unit)
    total_h = int((Y1 - Y0) * px_per_unit)
    print(f"    símbolos borrados: {stats['insert']} INSERT + {stats['circle']} CIRCLE "
          f"+ {stats['hatch']} HATCH + {sel_stats['inside_symbol']} entidades internas "
          f"(símbolo medio {symbol_cad:.3f} CAD)")
    print(f"    entidades de fondo conservadas: {sel_stats['kept']}")
    print(f"    render: {total_w}x{total_h}px @ {px_per_unit:.1f}px/unidad")

    # Ventanas de render, para no armar una imagen de cientos de megapíxeles.
    win = bg_cfg.render_window_px
    win_cad = win / px_per_unit
    # solapamos las ventanas un tile para no perder tiles a caballo del borde
    win_step_cad = (win - tile_size) / px_per_unit
    if win_step_cad <= 0:
        win_step_cad = win_cad

    candidates: list[np.ndarray] = []
    n_blank = n_symbol = 0

    ny = max(1, math.ceil((Y1 - Y0) / win_step_cad))
    nx = max(1, math.ceil((X1 - X0) / win_step_cad))
    if nx * ny > bg_cfg.max_render_windows:
        print(f"    ! {nx*ny} ventanas supera el máximo ({bg_cfg.max_render_windows}); "
              f"se muestrea aleatoriamente")

    windows = [(ix, iy) for iy in range(ny) for ix in range(nx)]
    rng.shuffle(windows)
    windows = windows[: bg_cfg.max_render_windows]

    for ix, iy in windows:
        wx0 = X0 + ix * win_step_cad
        wy0 = Y0 + iy * win_step_cad
        wx1, wy1 = wx0 + win_cad, wy0 + win_cad
        if wx0 >= X1 or wy0 >= Y1:
            continue

        img = render_window(doc, indexed, wx0, wy0, wx1, wy1, win, win)
        if img is None:
            continue
        H, W = img.shape[:2]

        gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        for ty in range(0, max(1, H - tile_size + 1), step):
            for tx in range(0, max(1, W - tile_size + 1), step):
                tile = img[ty:ty + tile_size, tx:tx + tile_size]
                if tile.shape[0] != tile_size or tile.shape[1] != tile_size:
                    continue
                gray = gray_full[ty:ty + tile_size, tx:tx + tile_size]
                ink = float(np.count_nonzero(gray < 200)) / gray.size
                # Un tile de fondo tiene que tener contenido real de plano,
                # no ser un margen en blanco con una línea suelta.
                if ink < bg_cfg.min_ink_fraction:
                    n_blank += 1
                    continue
                if float(np.std(tile)) < min_std_dev:
                    n_blank += 1
                    continue
                candidates.append(tile.copy())

        if len(candidates) >= max_tiles_per_plan * 3:
            break

    if not candidates:
        print(f"    -> 0 tiles útiles ({n_blank} con muy poco contenido)")
        return 0

    rng.shuffle(candidates)
    keep = candidates[:max_tiles_per_plan]
    for i, tile in enumerate(keep):
        out = output_dir / f"bg_{plano.stem}_{i:05d}.jpg"
        cv2.imwrite(str(out), tile, [cv2.IMWRITE_JPEG_QUALITY, 92])

    print(f"    -> {len(keep)} tiles guardados "
          f"({n_blank} descartados por tener poco contenido, "
          f"{len(candidates)} candidatos)")
    return len(keep)


if __name__ == "__main__":
    generate_backgrounds()

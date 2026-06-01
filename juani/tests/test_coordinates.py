"""
tests/test_coordinates.py

Tests para CoordinateMapper y CanvasMeta.

Por qué testear esto primero: es la lógica matemática más crítica del sistema.
Un bug aquí hace que todas las detecciones y specs queden en posiciones incorrectas,
y es exactamente el tipo de error que no se ve hasta comparar con AutoCAD.
"""
import pytest
from detection.coordinates import CanvasMeta, CoordinateMapper


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def simple_meta():
    """Canvas 100×100 px que mapea 1:1 con el espacio CAD 0-100."""
    return CanvasMeta(
        px_per_cad     = 1.0,
        x_min_cad      = 0.0,
        y_min_cad      = 0.0,
        x_max_cad      = 100.0,
        y_max_cad      = 100.0,
        image_width_px = 100,
        image_height_px= 100,
    )


@pytest.fixture
def scaled_meta():
    """Canvas 1000×500 px con factor de escala 10 px/CAD."""
    return CanvasMeta(
        px_per_cad     = 10.0,
        x_min_cad      = 0.0,
        y_min_cad      = 0.0,
        x_max_cad      = 100.0,
        y_max_cad      = 50.0,
        image_width_px = 1000,
        image_height_px= 500,
    )


# ── CanvasMeta ────────────────────────────────────────────────────────────────

def test_canvas_meta_from_dict(simple_meta):
    d = simple_meta.to_dict()
    reconstructed = CanvasMeta.from_dict(d)
    assert reconstructed == simple_meta


def test_canvas_meta_is_immutable(simple_meta):
    with pytest.raises((AttributeError, TypeError)):
        simple_meta.px_per_cad = 99.0   # frozen dataclass no permite mutación


# ── CoordinateMapper: px → CAD ────────────────────────────────────────────────

def test_origin_px_maps_to_top_left_cad(simple_meta):
    """Píxel (0, 0) es la esquina superior-izquierda → CAD (x_min, y_max)."""
    mapper = CoordinateMapper(simple_meta)
    x, y = mapper.center_px_to_cad(0.0, 0.0)
    assert x == pytest.approx(0.0)
    assert y == pytest.approx(100.0)   # Y se invierte: 0px → y_max_cad


def test_bottom_right_px_maps_to_cad_origin(simple_meta):
    """Píxel (100, 100) es la esquina inferior-derecha → CAD (x_max, y_min)."""
    mapper = CoordinateMapper(simple_meta)
    x, y = mapper.center_px_to_cad(100.0, 100.0)
    assert x == pytest.approx(100.0)
    assert y == pytest.approx(0.0)


def test_center_px_maps_to_center_cad(simple_meta):
    """El centro del canvas en px debe mapear al centro del espacio CAD."""
    mapper = CoordinateMapper(simple_meta)
    x, y = mapper.center_px_to_cad(50.0, 50.0)
    assert x == pytest.approx(50.0)
    assert y == pytest.approx(50.0)


def test_y_axis_inversion(simple_meta):
    """Y crece hacia abajo en px pero hacia arriba en CAD."""
    mapper = CoordinateMapper(simple_meta)
    _, y_top = mapper.center_px_to_cad(50.0, 0.0)    # arriba en imagen
    _, y_bot = mapper.center_px_to_cad(50.0, 100.0)  # abajo en imagen
    assert y_top > y_bot, "Y en CAD debe ser mayor cuando el píxel está más arriba"


def test_scale_factor_applied(scaled_meta):
    """Con px_per_cad=10, 100px en X equivalen a 10 CAD."""
    mapper = CoordinateMapper(scaled_meta)
    x, _ = mapper.center_px_to_cad(100.0, 0.0)
    assert x == pytest.approx(10.0)


# ── CoordinateMapper: bbox_px → CAD ──────────────────────────────────────────

def test_bbox_x_min_max_consistent(simple_meta):
    """x_min_cad < x_max_cad para cualquier bbox válido."""
    mapper = CoordinateMapper(simple_meta)
    x_min, _, x_max, _ = mapper.bbox_px_to_cad(10.0, 20.0, 40.0, 60.0)
    assert x_min < x_max


def test_bbox_y_min_max_consistent(simple_meta):
    """y_min_cad < y_max_cad para cualquier bbox válido."""
    mapper = CoordinateMapper(simple_meta)
    _, y_min, _, y_max = mapper.bbox_px_to_cad(10.0, 20.0, 40.0, 60.0)
    assert y_min < y_max


def test_bbox_center_matches_center_px(simple_meta):
    """El centro del bbox en CAD debe coincidir con la conversión del centro en px."""
    mapper = CoordinateMapper(simple_meta)
    x1, y1, x2, y2 = 20.0, 30.0, 60.0, 70.0
    cx_px = (x1 + x2) / 2.0
    cy_px = (y1 + y2) / 2.0

    bx_min, by_min, bx_max, by_max = mapper.bbox_px_to_cad(x1, y1, x2, y2)
    bx_center = (bx_min + bx_max) / 2.0
    by_center = (by_min + by_max) / 2.0

    exp_x, exp_y = mapper.center_px_to_cad(cx_px, cy_px)
    assert bx_center == pytest.approx(exp_x)
    assert by_center == pytest.approx(exp_y)


# ── CoordinateMapper: CAD → px (inversa) ──────────────────────────────────────

def test_roundtrip_px_cad_px(simple_meta):
    """Convertir px→CAD→px debe retornar el valor original."""
    mapper = CoordinateMapper(simple_meta)
    cx, cy = 37.5, 62.3
    x_cad, y_cad = mapper.center_px_to_cad(cx, cy)
    cx2, cy2 = mapper.cad_to_px(x_cad, y_cad)
    assert cx2 == pytest.approx(cx, abs=1e-9)
    assert cy2 == pytest.approx(cy, abs=1e-9)

"""
tests/test_spec_extractor.py

Tests para SpecExtractor._assign_specs() y _collect_texts().

Por qué testear esto: el algoritmo de asignación de specs tiene lógica
geométrica sutil (banda vertical, gap-cutting, x_limit por componente vecino)
que puede silenciosamente asignar specs incorrectos sin error visible.
"""
import pytest
from detection.coordinates import CanvasMeta, CoordinateMapper
from extraction.spec_extractor import SpecExtractor


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def canvas():
    """Canvas sencillo: 1000×500 px, 10 px/CAD, origen en (0,0)."""
    return CanvasMeta(
        px_per_cad     = 10.0,
        x_min_cad      = 0.0,
        y_min_cad      = 0.0,
        x_max_cad      = 100.0,
        y_max_cad      = 50.0,
        image_width_px = 1000,
        image_height_px= 500,
    )


def make_det(bbox_px, clase="comp", conf=0.9):
    x1, y1, x2, y2 = bbox_px
    return {
        "clase":    clase,
        "conf":     conf,
        "bbox_px":  list(bbox_px),
        "centro_px":[(x1+x2)/2, (y1+y2)/2],
        "x_cad":    0.0, "y_cad": 0.0,
    }


def make_text(x, y, texto):
    return {"x": x, "y": y, "texto": texto}


# ── Tests de _assign_specs ─────────────────────────────────────────────────────

def test_single_text_in_band_is_captured(canvas):
    """Un texto a la derecha del componente dentro de la banda Y debe capturarse."""
    mapper = CoordinateMapper(canvas)
    # Componente en CAD: x=[2,4], y=[3,5] → px: x=[20,40], y=[450,470] (invertido)
    # y en px = (y_max_cad - y_cad) * px_per_cad → y=3 → (50-3)*10=470, y=5 → (50-5)*10=450
    det     = make_det(bbox_px=(20, 450, 40, 470))
    # Texto en CAD x=6, y=4 → dentro de la banda y=[3,5] y a la derecha del centro (x=3)
    texts   = [make_text(x=6.0, y=4.0, texto="25A")]

    specs = SpecExtractor._assign_specs(det, [det], texts, CoordinateMapper(canvas))
    assert specs == ["25A"]


def test_text_left_of_component_ignored(canvas):
    """Texto a la izquierda del centro del componente no se captura."""
    det   = make_det(bbox_px=(200, 450, 400, 470))   # centro x_cad ≈ 30
    # Texto en CAD x=5, que es menor que x_start (≈30)
    texts = [make_text(x=5.0, y=4.0, texto="ignorar")]
    specs = SpecExtractor._assign_specs(det, [det], texts, CoordinateMapper(canvas))
    assert specs == []


def test_text_outside_y_band_ignored(canvas):
    """Texto fuera de la banda vertical [y_min, y_max] no se captura."""
    det   = make_det(bbox_px=(20, 450, 40, 470))     # y_cad ≈ [3, 5]
    texts = [make_text(x=6.0, y=8.0, texto="fuera")]  # y=8 > y_max=5
    specs = SpecExtractor._assign_specs(det, [det], texts, CoordinateMapper(canvas))
    assert specs == []


def test_gap_cutting_stops_at_large_gap(canvas):
    """Si el gap entre textos es mayor que la altura del componente, se corta."""
    # Componente: bbox_px y≈[440,490] → alto_cad ≈ 5 CAD
    det   = make_det(bbox_px=(20, 440, 40, 490))
    texts = [
        make_text(x=5.0, y=4.0, texto="30mA"),   # cerca
        make_text(x=6.0, y=4.0, texto="2x25A"),  # gap=1, < alto=5 → incluir
        make_text(x=15.0, y=4.0, texto="UPS"),   # gap=9, > alto=5 → CORTAR
        make_text(x=16.0, y=4.0, texto="8kVA"),  # no debe llegar aquí
    ]
    specs = SpecExtractor._assign_specs(det, [det], texts, CoordinateMapper(canvas))
    assert "30mA" in specs
    assert "2x25A" in specs
    assert "UPS" not in specs
    assert "8kVA" not in specs


def test_x_limit_from_neighboring_component(canvas):
    """El texto entre dos componentes sólo se asigna al de la izquierda."""
    # Componente A: x_cad ≈ [2, 4]
    det_a = make_det(bbox_px=(20, 450, 40, 470), clase="A")
    # Componente B: x_cad ≈ [8, 10], en la misma fila
    det_b = make_det(bbox_px=(80, 450, 100, 470), clase="B")
    # Texto en x_cad=6, dentro de la banda de A y antes de B
    texts = [make_text(x=6.0, y=4.0, texto="spec_de_A")]

    specs_a = SpecExtractor._assign_specs(det_a, [det_a, det_b], texts, CoordinateMapper(canvas))
    specs_b = SpecExtractor._assign_specs(det_b, [det_a, det_b], texts, CoordinateMapper(canvas))

    assert "spec_de_A" in specs_a
    assert "spec_de_A" not in specs_b



def test_no_texts_returns_empty(canvas):
    det   = make_det(bbox_px=(20, 450, 40, 470))
    specs = SpecExtractor._assign_specs(det, [det], [], CoordinateMapper(canvas))
    assert specs == []


def test_extract_from_doc_enriches_detections(canvas, tmp_path):
    """Test de integración liviano: extract_from_doc agrega campo 'specs'."""
    import ezdxf

    # Crear DXF mínimo en memoria
    doc = ezdxf.new()
    msp = doc.modelspace()
    msp.add_text("25A", dxfattribs={"insert": (6.0, 4.0), "height": 0.5})

    det = make_det(bbox_px=(20, 450, 40, 470))

    extractor = SpecExtractor()
    result    = extractor.extract_from_doc(doc, [det], canvas)

    assert len(result) == 1
    assert "specs" in result[0]
    assert isinstance(result[0]["specs"], list)

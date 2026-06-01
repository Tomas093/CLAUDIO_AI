"""
tests/test_postprocess.py

Tests para los filtros de post-proceso y PostProcessorChain.

Por qué testear esto: cada filtro tiene lógica de umbral/geometría
que puede romperse silenciosamente si se cambian los thresholds en config.
Los tests documentan el comportamiento esperado para cada caso límite.
"""
import pytest
from detection.postprocess import (
    BorderFilter,
    AgnosticNMSFilter,
    NestedBoxFilter,
    ConfidenceFilter,
    PostProcessorChain,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_det(clase="comp", conf=0.8, bbox=(10, 10, 50, 50)):
    x1, y1, x2, y2 = bbox
    return {
        "clase":    clase,
        "conf":     conf,
        "bbox_px":  list(bbox),
        "centro_px":[(x1 + x2) / 2, (y1 + y2) / 2],
        "x_cad":    0.0,
        "y_cad":    0.0,
    }


SAMPLE_META = {
    "image_width_px":  1000,
    "image_height_px": 800,
}


# ── ConfidenceFilter ──────────────────────────────────────────────────────────

def test_confidence_filter_removes_low():
    dets = [make_det(conf=0.8), make_det(conf=0.3), make_det(conf=0.6)]
    result = ConfidenceFilter(conf_min=0.5).filter(dets)
    assert len(result) == 2
    assert all(d["conf"] >= 0.5 for d in result)


def test_confidence_filter_keeps_all_above_threshold():
    dets = [make_det(conf=0.9), make_det(conf=0.7), make_det(conf=0.51)]
    result = ConfidenceFilter(conf_min=0.5).filter(dets)
    assert len(result) == 3


def test_confidence_filter_zero_disables():
    dets = [make_det(conf=0.01), make_det(conf=0.0)]
    result = ConfidenceFilter(conf_min=0.0).filter(dets)
    assert len(result) == 2   # conf_min=0 no filtra nada


def test_confidence_filter_empty_input():
    assert ConfidenceFilter(0.5).filter([]) == []


# ── BorderFilter ──────────────────────────────────────────────────────────────

def test_border_filter_removes_touching_top():
    det = make_det(conf=0.6, bbox=(10, 0, 50, 40))   # y1=0, toca borde superior
    result = BorderFilter(margin_px=2, conf_safe=0.9).filter([det], meta=SAMPLE_META)
    assert len(result) == 0


def test_border_filter_keeps_high_conf_on_border():
    det = make_det(conf=0.95, bbox=(10, 0, 50, 40))  # toca borde pero conf alta
    result = BorderFilter(margin_px=2, conf_safe=0.9).filter([det], meta=SAMPLE_META)
    assert len(result) == 1


def test_border_filter_keeps_interior_dets():
    det = make_det(conf=0.5, bbox=(50, 50, 200, 200))  # lejos del borde
    result = BorderFilter(margin_px=2, conf_safe=0.9).filter([det], meta=SAMPLE_META)
    assert len(result) == 1


def test_border_filter_right_edge():
    det = make_det(conf=0.5, bbox=(950, 100, 1000, 200))  # x2=1000 = width
    result = BorderFilter(margin_px=2, conf_safe=0.9).filter([det], meta=SAMPLE_META)
    assert len(result) == 0


# ── AgnosticNMSFilter ─────────────────────────────────────────────────────────

def test_nms_removes_duplicate():
    """Dos cajas casi idénticas → debe quedar la de mayor conf."""
    d1 = make_det(conf=0.9, bbox=(10, 10, 60, 60))
    d2 = make_det(conf=0.5, bbox=(12, 12, 62, 62))   # IoU alto con d1
    result = AgnosticNMSFilter(iou_thresh=0.5).filter([d1, d2])
    assert len(result) == 1
    assert result[0]["conf"] == pytest.approx(0.9)


def test_nms_keeps_non_overlapping():
    d1 = make_det(conf=0.9, bbox=(0,   0,  50,  50))
    d2 = make_det(conf=0.8, bbox=(200, 200, 250, 250))  # sin overlap
    result = AgnosticNMSFilter(iou_thresh=0.5).filter([d1, d2])
    assert len(result) == 2


def test_nms_agnostic_across_classes():
    """NMS debe suprimir cajas solapadas aunque sean de clases distintas."""
    d1 = make_det(clase="termomagnetico", conf=0.9, bbox=(10, 10, 60, 60))
    d2 = make_det(clase="diferencial",   conf=0.5, bbox=(12, 12, 62, 62))
    result = AgnosticNMSFilter(iou_thresh=0.5).filter([d1, d2])
    assert len(result) == 1


def test_nms_empty_input():
    assert AgnosticNMSFilter().filter([]) == []


# ── NestedBoxFilter ───────────────────────────────────────────────────────────

def test_nested_removes_contained_box():
    """Caja pequeña totalmente dentro de caja grande → se elimina la pequeña."""
    big   = make_det(conf=0.9, bbox=(0,  0,  100, 100))
    small = make_det(conf=0.5, bbox=(10, 10, 40,  40))   # dentro de big
    result = NestedBoxFilter(ios_thresh=0.7).filter([big, small])
    assert len(result) == 1
    assert result[0]["conf"] == pytest.approx(0.9)


def test_nested_keeps_partially_overlapping():
    """Cajas que se solapan parcialmente (IOS bajo) → ambas sobreviven."""
    d1 = make_det(conf=0.9, bbox=(0,  0,  60, 60))
    d2 = make_det(conf=0.7, bbox=(40, 40, 100, 100))   # overlap parcial
    result = NestedBoxFilter(ios_thresh=0.7).filter([d1, d2])
    assert len(result) == 2


def test_nested_winner_is_higher_conf():
    """Si hay anidamiento, gana la de mayor confianza."""
    inner = make_det(conf=0.95, bbox=(20, 20, 50, 50))
    outer = make_det(conf=0.60, bbox=(10, 10, 60, 60))  # outer tiene menor conf
    result = NestedBoxFilter(ios_thresh=0.7).filter([inner, outer])
    assert len(result) == 1
    assert result[0]["conf"] == pytest.approx(0.95)


# ── PostProcessorChain ────────────────────────────────────────────────────────

def test_chain_applies_filters_in_order():
    """La cadena debe aplicar todos los filtros secuencialmente."""
    # Solo el det con conf=0.9 y bbox interior debe sobrevivir
    dets = [
        make_det(conf=0.9, bbox=(100, 100, 200, 200)),
        make_det(conf=0.3, bbox=(300, 300, 400, 400)),  # eliminado por conf_min
        make_det(conf=0.7, bbox=(0, 0, 50, 50)),        # eliminado por borde
    ]
    chain = PostProcessorChain([
        BorderFilter(margin_px=2, conf_safe=0.95),
        ConfidenceFilter(conf_min=0.5),
    ])
    result = chain.run(dets, meta=SAMPLE_META)
    assert len(result) == 1
    assert result[0]["conf"] == pytest.approx(0.9)


def test_chain_with_no_filters_is_passthrough():
    dets = [make_det(), make_det()]
    chain = PostProcessorChain([])
    result = chain.run(dets, meta=SAMPLE_META)
    assert len(result) == 2


def test_chain_empty_input():
    chain = PostProcessorChain([ConfidenceFilter(0.5), BorderFilter()])
    result = chain.run([], meta=SAMPLE_META)
    assert result == []

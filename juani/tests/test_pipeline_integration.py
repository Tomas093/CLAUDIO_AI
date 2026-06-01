"""
tests/test_pipeline_integration.py

Tests de integración del pipeline completo.

Estrategia: mockear DetectionModel.load() para no necesitar GPU ni pesos .pt reales.
El DXF mínimo se genera programáticamente con ezdxf.
Esto permite verificar que el pipeline orquesta correctamente las etapas sin
depender de infraestructura externa.
"""
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

import ezdxf

from config import PipelineConfig


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def minimal_dxf(tmp_path) -> Path:
    """DXF mínimo con un bloque INSERT y un texto."""
    doc = ezdxf.new()
    msp = doc.modelspace()

    # Bloque simple (para que InsertScaleStrategy encuentre algo)
    blk = doc.blocks.new("SYM_TEST")
    blk.add_line((0, 0, 0), (2, 0, 0))
    blk.add_line((0, 0, 0), (0, 2, 0))
    msp.add_blockref("SYM_TEST", (5.0, 5.0))
    msp.add_blockref("SYM_TEST", (10.0, 5.0))

    # Texto de spec
    msp.add_text("25A", dxfattribs={"insert": (8.0, 5.5), "height": 0.5})

    dxf_path = tmp_path / "test_minimal.dxf"
    doc.saveas(str(dxf_path))
    return dxf_path


@pytest.fixture
def base_cfg(minimal_dxf, tmp_path) -> PipelineConfig:
    return PipelineConfig(
        dxf_path    = str(minimal_dxf),
        output_dir  = str(tmp_path / "out"),
        model_paths = ["dummy_model.pt"],
        device      = "cpu",
    )


def make_mock_detection():
    """Detección falsa que simula la salida de ImageSlicer.run_batched."""
    return {
        "clase":    "termomagnetico",
        "conf":     0.85,
        "bbox_px":  [100.0, 100.0, 200.0, 200.0],
        "centro_px":[150.0, 150.0],
        "x_cad":    5.0,
        "y_cad":    5.0,
    }


# ── Tests de integración ──────────────────────────────────────────────────────

def test_pipeline_run_produces_output_files(base_cfg):
    """El pipeline debe generar los archivos de salida esperados."""
    with patch("pipeline.base.DetectionModel") as MockModel, \
         patch("detection.slicer.ImageSlicer.run_batched") as mock_slicer:

        mock_slicer.return_value = [make_mock_detection()]

        from pipeline.base import Pipeline
        pipeline = Pipeline(base_cfg)
        conteo, dets = pipeline.run()

    out = Path(base_cfg.output_dir)
    assert (out / "plano_render.png").exists(),         "falta plano_render.png"
    assert (out / "plano_render_viz.png").exists(),     "falta plano_render_viz.png"
    assert (out / "deteccion_visual.png").exists(),     "falta deteccion_visual.png"
    assert (out / "detecciones.json").exists(),         "falta detecciones.json"
    assert (out / "detecciones_con_specs.json").exists(),"falta detecciones_con_specs.json"


def test_pipeline_detections_json_format(base_cfg):
    """Las detecciones guardadas deben tener el schema correcto."""
    with patch("pipeline.base.DetectionModel"), \
         patch("detection.slicer.ImageSlicer.run_batched") as mock_slicer:

        mock_slicer.return_value = [make_mock_detection()]

        from pipeline.base import Pipeline
        pipeline = Pipeline(base_cfg)
        pipeline.run()

    with open(Path(base_cfg.output_dir) / "detecciones_con_specs.json") as f:
        dets = json.load(f)

    assert len(dets) >= 1
    for d in dets:
        assert "clase"    in d
        assert "conf"     in d
        assert "bbox_px"  in d
        assert "specs"    in d
        assert isinstance(d["specs"], list)


def test_pipeline_empty_detections(base_cfg):
    """Pipeline con 0 detecciones no debe lanzar excepción."""
    with patch("pipeline.base.DetectionModel"), \
         patch("detection.slicer.ImageSlicer.run_batched") as mock_slicer:

        mock_slicer.return_value = []   # modelo no detecta nada

        from pipeline.base import Pipeline
        conteo, dets = Pipeline(base_cfg).run()

    assert conteo == {}
    assert dets == []


def test_ensemble_pipeline_aggregates_models(base_cfg):
    """EnsemblePipeline debe correr N modelos y combinar sus detecciones."""
    base_cfg.model_paths = ["model_a.pt", "model_b.pt"]

    call_count = 0

    def fake_run_batched(model, image_path, meta):
        nonlocal call_count
        call_count += 1
        return [make_mock_detection()]

    with patch("pipeline.base.DetectionModel.load") as mock_load, \
         patch("detection.slicer.ImageSlicer.run_batched", side_effect=fake_run_batched):

        mock_load.return_value = MagicMock()

        from pipeline.ensemble import EnsemblePipeline
        EnsemblePipeline(base_cfg).run()

    assert call_count == 2, f"Se esperaban 2 llamadas a run_batched, hubo {call_count}"


def test_meta_json_has_required_fields(base_cfg):
    """El JSON de metadata del render debe contener los campos necesarios para CoordinateMapper."""
    with patch("pipeline.base.DetectionModel"), \
         patch("detection.slicer.ImageSlicer.run_batched") as mock_slicer:

        mock_slicer.return_value = []

        from pipeline.base import Pipeline
        Pipeline(base_cfg).run()

    meta_path = Path(base_cfg.output_dir) / "plano_render.json"
    assert meta_path.exists()

    with open(meta_path) as f:
        meta = json.load(f)

    required = ["px_per_cad", "x_min_cad", "y_min_cad", "x_max_cad",
                "y_max_cad", "image_width_px", "image_height_px"]
    for field in required:
        assert field in meta, f"Campo '{field}' faltante en meta JSON"

# train-maker

Pipeline separada para generar datos sintéticos YOLO a partir de un DXF y fondos reales.

## Archivos

- `config.py`: rutas y parámetros.
- `phase1_extractor.py`: render y variaciones de sprites.
- `phase2_3_fusion_labeler.py`: composición sintética y etiquetas YOLO.
- `phase4_assembler.py`: split train/val, negativos y `data.yaml`.
- `run_pipeline.py`: orquestador principal.
- `ib_maker.py`: compatibilidad con la versión anterior.

## Uso

```bash
cd /train-maker
pip install -r requirements.txt
python run_pipeline.py
```

## Entradas esperadas

- `input/component.dxf`
- `input/backgrounds/`

## Salidas

- `output/sprites/`
- `output/synthetic/`
- `dataset/`


@echo off
echo ============================================================
echo  Entrenando modelos single-clase (7 clases)
echo ============================================================

echo.
echo [1/7] ojo_de_buey
yolo train model=yolov8n.pt data=datasets_out/single_ojo_de_buey/data.yaml epochs=50 imgsz=640 batch=8 device=0 name=synth_single_ojo_de_buey workers=0

echo.
echo [2/7] interruptor_termomagnetico
yolo train model=yolov8n.pt data=datasets_out/single_interruptor_termomagnetico/data.yaml epochs=50 imgsz=640 batch=8 device=0 name=synth_single_termomagnetico workers=0

echo.
echo [3/7] interruptor_diferencial
yolo train model=yolov8n.pt data=datasets_out/single_interruptor_diferencial/data.yaml epochs=50 imgsz=640 batch=8 device=0 name=synth_single_diferencial workers=0

echo.
echo [4/7] interruptor_seleccionador_manual
yolo train model=yolov8n.pt data=datasets_out/single_interruptor_seleccionador_manual/data.yaml epochs=50 imgsz=640 batch=8 device=0 name=synth_single_seleccionador_manual workers=0

echo.
echo [5/7] seleccionador_bajo_carga
yolo train model=yolov8n.pt data=datasets_out/single_seleccionador_bajo_carga/data.yaml epochs=50 imgsz=640 batch=8 device=0 name=synth_single_seleccionador_bajo_carga workers=0

echo.
echo [6/7] multi_medidor
yolo train model=yolov8n.pt data=datasets_out/single_multi_medidor/data.yaml epochs=50 imgsz=640 batch=8 device=0 name=synth_single_multi_medidor workers=0

echo.
echo [7/7] interruptor_motorizado
yolo train model=yolov8n.pt data=datasets_out/single_interruptor_motorizado/data.yaml epochs=50 imgsz=640 batch=8 device=0 name=synth_single_interruptor_motorizado workers=0

echo.
echo ============================================================
echo  Todos los modelos entrenados. Pesos en:
echo  runs/detect/synth_single_*/weights/best.pt
echo ============================================================
pause

@echo off
echo ============================================================
echo  Fine-tuning modelos single-clase sobre datos reales
echo  Base: runs/detect/synth_single_*/weights/best.pt
echo  Datos: png-dataset-remapped/single_*/
echo ============================================================

echo.
echo [1/7] ojo_de_buey  (48 train / 15 val reales)
yolo train model=runs/detect/synth_single_ojo_de_buey/weights/best.pt data=png-dataset-remapped/single_ojo_de_buey/data.yaml epochs=30 imgsz=640 batch=8 device=0 name=ft_single_ojo_de_buey workers=0 lr0=0.001 lrf=0.01

echo.
echo [2/7] interruptor_termomagnetico  (248 train / 86 val reales)
yolo train model=runs/detect/synth_single_termomagnetico/weights/best.pt data=png-dataset-remapped/single_interruptor_termomagnetico/data.yaml epochs=30 imgsz=640 batch=8 device=0 name=ft_single_termomagnetico workers=0 lr0=0.001 lrf=0.01

echo.
echo [3/7] interruptor_diferencial  (241 train / 87 val reales)
yolo train model=runs/detect/synth_single_diferencial/weights/best.pt data=png-dataset-remapped/single_interruptor_diferencial/data.yaml epochs=30 imgsz=640 batch=8 device=0 name=ft_single_diferencial workers=0 lr0=0.001 lrf=0.01

echo.
echo [4/7] interruptor_seleccionador_manual  (109 train / 34 val reales)
yolo train model=runs/detect/synth_single_seleccionador_manual/weights/best.pt data=png-dataset-remapped/single_interruptor_seleccionador_manual/data.yaml epochs=30 imgsz=640 batch=8 device=0 name=ft_single_seleccionador_manual workers=0 lr0=0.001 lrf=0.01

echo.
echo [5/7] seleccionador_bajo_carga  (3 train / 4 val reales - pocas muestras)
yolo train model=runs/detect/synth_single_seleccionador_bajo_carga/weights/best.pt data=png-dataset-remapped/single_seleccionador_bajo_carga/data.yaml epochs=15 imgsz=640 batch=4 device=0 name=ft_single_seleccionador_bajo_carga workers=0 lr0=0.0005 lrf=0.01

echo.
echo [6/7] multi_medidor  (6 train / 0 val reales - pocas muestras)
yolo train model=runs/detect/synth_single_multi_medidor/weights/best.pt data=png-dataset-remapped/single_multi_medidor/data.yaml epochs=15 imgsz=640 batch=4 device=0 name=ft_single_multi_medidor workers=0 lr0=0.0005 lrf=0.01

echo.
echo [7/7] interruptor_motorizado  (5 train / 0 val reales - pocas muestras)
yolo train model=runs/detect/synth_single_interruptor_motorizado/weights/best.pt data=png-dataset-remapped/single_interruptor_motorizado/data.yaml epochs=15 imgsz=640 batch=4 device=0 name=ft_single_interruptor_motorizado workers=0 lr0=0.0005 lrf=0.01

echo.
echo ============================================================
echo  Fine-tuning completo. Pesos finales en:
echo    runs/detect/ft_single_*/weights/best.pt
echo ============================================================
echo.
echo  Ensemble: usa estos 7 modelos juntos con ensemble_predict.py
pause

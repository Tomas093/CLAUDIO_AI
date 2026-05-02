from ultralytics import YOLO
import torch

def train_model():
    # 1. Cargar el modelo base (YOLOv8 nano es ideal por velocidad y precisión en símbolos)
    model = YOLO('yolov8n.pt')

    # 2. Verificar si CUDA está disponible para usar tu 3060 Ti
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"Entrenando en: {device}")

    # 3. Lanzar el entrenamiento
    model.train(
        data='dataset/data.yaml',    # Ruta al archivo que generó la Fase 4
        epochs=100,                  # 100 épocas es un buen punto de partida
        imgsz=640,                   # Tamaño de imagen estándar
        batch=32,                    # Ajustalo según la memoria de tu GPU (8, 16 o 32)
        name='liard_interruptor_v1', # Nombre de la carpeta de salida
        device=device,               # Usar tu GPU NVIDIA
        patience=20,                 # Early stopping: para si no mejora en 20 épocas
        augment=True                 # Activa el Data Augmentation interno de YOLO
    )

if __name__ == '__main__':
    train_model()
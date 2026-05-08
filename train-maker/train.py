from ultralytics import YOLO
import torch

def train_model():
    # Usar el modelo 'medium' en lugar de 'nano'
    model = YOLO('yolov8n.pt')

    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"Entrenando en: {device}")

    model.train(
        # ACÁ ESTÁ EL ARREGLO: Ruta absoluta con una "r" antes de las comillas
        data=r'C:\Users\Tomas\Documents\LAB3\CLAUDIO_AI\train-maker\dataset',
        epochs=100,
        imgsz=640,
        batch=32,
        name='liard_multiclase_v1',
        device=device,
        patience=20,
        augment=True
    )

if __name__ == '__main__':
    print("CUDA disponible:", torch.cuda.is_available())
    print("Versión CUDA de PyTorch:", torch.version.cuda)
    print("Cantidad de GPUs:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    train_model()
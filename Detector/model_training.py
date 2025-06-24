from ultralytics import YOLO
import torch

torch.cuda.empty_cache()  # Clear GPU memory

detector = YOLO("yolov8m.pt")  # Load a pretrained YOLOv8 model

# Training the model
train_results = detector.train(data="RPG-Die.v6i.yolov8/data.yaml", epochs=60, imgsz=608, name="v8m_dice", batch=0.8, cache=True)
# Save the model

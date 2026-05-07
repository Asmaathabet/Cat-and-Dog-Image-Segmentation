from ultralytics import YOLO
# Load a model
model = YOLO('yolo11n-seg.pt')  # load a pretrained model (recommended for training)
# Train the model
model.train(data='custom-dataset.yaml', epochs=100, imgsz=640, batch=8, device='cpu')

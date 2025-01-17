from ultralytics import YOLO
model = YOLO('yolov8n.pt')

model.train(data='path.yaml', epochs=100, imgsz=640)
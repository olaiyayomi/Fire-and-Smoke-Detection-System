from ultralytics import YOLO

model = YOLO("best.pt")

result = model.predict("data/firesmoke2.mp4", show=True, save=True)
from ultralytics import YOLO

# Camera prediction
model = YOLO("yolov8n.pt")
resulte = model.predict(source=0, show=True, verbose=False)
print(resulte)
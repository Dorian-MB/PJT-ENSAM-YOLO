from ultralytics import YOLO

# Camera prediction
model = YOLO("models/best_s.pt")
# model = YOLO("yolov8s.pt")

resulte = model.predict(source=0, show=True, verbose=False)

# print(resulte)
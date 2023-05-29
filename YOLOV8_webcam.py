from ultralytics import YOLO

# Prediction
model_path = "models/nano_189epochs_SGD.pt"
model = YOLO(model_path)

source = "input/video_test_5.mp4"
# source = 0 # for live detection (webcam needed)
result = model.predict(source=source, save=True)
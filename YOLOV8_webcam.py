from ultralytics import YOLO

# Prediction
model_path = "models/small_141epochs_SGD.pt"
model = YOLO(model_path)

source = "input/test_outlet_03.mp4" # 
# source = 0 # for live detection (webcam needed)
result = model.predict(source=source, save=True)
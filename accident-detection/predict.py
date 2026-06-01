from ultralytics import YOLO
import sys

model = YOLO("best.pt")

image_path = sys.argv[1]

model.predict(
    source=image_path,
    conf=0.25,
    save=True
)

print("Prediction completed")
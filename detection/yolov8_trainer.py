ROOT_DIR='path/to/your/dataset/directory'

import os
from ultralytics import YOLO


# # Load a model
model = YOLO("yolov8n.yaml")

# # Use the model
results=model.train(data=os.path.join(ROOT_DIR, "config.yaml"), epochs=100, resume=True, conf=0.25, iou=0.5)

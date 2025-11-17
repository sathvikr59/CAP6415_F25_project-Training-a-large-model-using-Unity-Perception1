from ultralytics import YOLO
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--weights', required=True, help="Path to trained weights")
parser.add_argument('--data', required=True, help="Path to YAML file")
args = parser.parse_args()

# Load trained YOLOv8 model
model = YOLO(args.weights)

# Evaluate model on validation set
results = model.val(data=args.data)
print(results)

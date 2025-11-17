from ultralytics import YOLO
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--data', required=True, help="Path to YAML file")
parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs")
args = parser.parse_args()

os.makedirs('results', exist_ok=True)

# Load YOLOv8 pre-trained model
model = YOLO('yolov8m.pt')

# Start training
model.train(
    data=args.data,
    epochs=args.epochs,
    imgsz=640,
    batch=16,
    project='results',
    name='experiment'
)

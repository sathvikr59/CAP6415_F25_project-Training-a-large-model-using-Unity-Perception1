from ultralytics import YOLO
import argparse, os
parser = argparse.ArgumentParser()
parser.add_argument('--data', required=True)
parser.add_argument('--epochs', type=int, default=50)
args = parser.parse_args()
os.makedirs('results', exist_ok=True)
model = YOLO('yolov8m.pt')
model.train(data=args.data, epochs=args.epochs, imgsz=640, batch=16, project='results')

from ultralytics import YOLO
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--weights', required=True)
parser.add_argument('--data', required=True)
args = parser.parse_args()
model = YOLO(args.weights)
results = model.val(data=args.data)
print(results)

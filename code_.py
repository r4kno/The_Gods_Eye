import cv2
from ultralytics import YOLO
import torch

#Standard

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

model = YOLO('yolov8m.pt').to(device)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
desired_classes = [0]
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image")
        break

    results = model.track(frame,persist=True, classes=desired_classes)

    annotated_frame = results[0].plot()
    
    cv2.imshow('YOLOv8 Webcam Detection', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
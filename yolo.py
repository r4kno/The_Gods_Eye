import cv2
from ultralytics import YOLO
import torch
import threading

#Multi-threading


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

def multiThreadTacking (video_input, model, file_index):


    cap = cv2.VideoCapture(video_input)
    print("Tracking the video feed: ", file_index)
    
    if not cap.isOpened():
        print("Error: Could not open video input.")
        exit()

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break

        results = model.track(frame, persist=True)

        annotated_frame = results[0].plot()
        
        cv2.imshow('YOLOv8 Webcam Detection', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()



model1 = YOLO('yolov8m.pt').to(device)

video_file1 = 0
video_file2 = 1

tracker_thread1 = threading.Thread(target=multiThreadTacking, args=(video_file1, model1, 1), daemon=True)
tracker_thread2 = threading.Thread(target=multiThreadTacking, args=(video_file2, model1, 2), daemon=True)

tracker_thread1.start()
tracker_thread2.start()

tracker_thread1.join()
tracker_thread2.join()

cv2.destroyAllWindows()
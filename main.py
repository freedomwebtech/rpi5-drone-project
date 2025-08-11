import cv2
from ultralytics import YOLO
import numpy as np
from picamera2 import Picamera2
import time
import cvzone
# Load YOLOv8 model
model = YOLO('best_float32.tflite')
names = model.names

# Initialize Picamera2
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()
time.sleep(2)  # Let camera warm up

hist={}
line_y=332
count=0

frame_count = 0

while True:
    # Capture frame from Pi Camera
    frame = picam2.capture_array()
    frame = cv2.flip(frame, -1)  # Flip vertically and horizontally if needed

    frame_count += 1
    if frame_count % 3 != 0:
        continue

    frame = cv2.resize(frame, (1020,500))
    results = model.track(frame, persist=True,imgsz=240)

    if results[0].boxes.id is not None:
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        class_ids = results[0].boxes.cls.int().cpu().tolist()

        for track_id, box, class_id in zip(ids, boxes, class_ids):
            x1, y1, x2, y2 = box
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            name = names[class_id]
            cv2.rectangle(frame, (x1, y1), (x2, y2),(0,255,0), 2)
            cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
            cv2.putText(frame, f"{name} ID:{track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            if name == 'cabbage':
                cv2.rectangle(frame, (x1, y1), (x2, y2),(0,255,0), 2)
                cv2.circle(frame,(cx,cy),4,(255,0,0),-1)
                if track_id in hist:
                   prev_cx,prev_cy=hist[track_id]
                   if prev_cy<line_y<=cy:
                      count+=1
                      cv2.rectangle(frame, (x1, y1), (x2, y2),(0,0,255), 2)
                      cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
                      cv2.putText(frame, f"{name} ID:{track_id}", (x1, y1 - 10),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            hist[track_id]=(cx,cy)       
         

            
    cvzone.putTextRect(frame, f'COUNT-: {count}', (50, 30), 2, 2, colorR=(255, 0, 255))
    cv2.line(frame,(0,line_y),(frame.shape[1],line_y),(255,0,0),2)

    cv2.imshow("FRAME", frame)

    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

picam2.close()
cv2.destroyAllWindows()



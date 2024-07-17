import cv2 as cv
import torch
from ultralytics import YOLO

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

yolo = YOLO('model/yolov8l-face.pt')

cam = cv.VideoCapture(0)

while True:
    ret, frame = cam.read()

    if ret:
        frame = cv.flip(frame, 1)

        results = yolo.predict(frame, conf=0.8)

        if len(results)>0:
            boxes = results[0].boxes.xyxy.tolist()
            
            for box in boxes:
                pt1 = (int(box[0]), int(box[1]))
                pt2 = (int(box[2]), int(box[3]))

                cv.rectangle(frame, pt1, pt2, (0,255,0), 1)

        cv.imshow('frame', frame)
        key = cv.waitKey(1)
        if key == ord('q'):
            break
    else:
        break

cam.release()
cv.destroyAllWindows()
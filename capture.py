import cv2 as cv
import torch
from ultralytics import YOLO
import os

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

yolo = YOLO('model/yolov8l-face.pt')
yolo.to(device)



def FaceDetector(img):
    global yolo
    bounding_box = None

    results = yolo.predict(img, conf=0.7)
    if len(results) > 0:
        boxes = results[0].boxes.xyxy.tolist()
        box = boxes[0]
        pt1 = (int(box[0]), int(box[1]))
        pt2 = (int(box[2]), int(box[3]))
        bounding_box = (pt1, pt2)

    return bounding_box

def FindFocalLength(ref_image, known_width, known_distance):
    bounding_box = FaceDetector(ref_image)
    focal_length = None
    if bounding_box == None:
        pass
    else:
        pt1, pt2 = bounding_box
        pixel_width = pt2[0] - pt1[0]

        focal_length = (known_distance * pixel_width) / known_width

    return focal_length

def DistanceEstimation(focal_length, known_width, pixel_width):
    return round((focal_length*known_width)/pixel_width, 2)

if __name__ == '__main__':    
    cam = cv.VideoCapture(0)

    known_width = 14.3 #cm
    known_distance = 76.2 #cm
    ref_img_path = 'Ref_image.png'
    ref_img = cv.imread(ref_img_path) 
    focal_length = FindFocalLength(ref_img, known_width, known_distance)
    
    os.chdir('data/images/depth_train')
    filename = ''

    assert focal_length is not None, 'No reference face'

    while True:
        ret, frame = cam.read()

        if ret:
            frame = cv.flip(frame, 1)
            output = frame.copy()
            results = yolo.predict(frame, conf=0.8)

            if len(results)>0:
                boxes = results[0].boxes.xyxy.tolist()

                for box in boxes:
                    pt1 = (int(box[0]), int(box[1]))
                    pt2 = (int(box[2]), int(box[3]))
                    pixel_width = pt2[0] - pt1[0]
                    distance = DistanceEstimation(focal_length, known_width, pixel_width)
                    cv.rectangle(frame, pt1, pt2, (0,255,0), 1)
                    cv.putText(frame, f'Distance estimated: {distance} cm', (pt1[0], pt1[1]-10), cv.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0), 1)

            cv.imshow('frame', frame)
            key = cv.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('w'):
                filename = str(input('Nhap ten file:'))
            elif key == ord('c'):
                if len(filename) == 0:
                    print('Nhap ten file')
                else:
                    cv.imwrite(filename, output)
        else:
            break
        
    cam.release()
    cv.destroyAllWindows()
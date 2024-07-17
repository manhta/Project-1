import torch
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
from ultralytics import YOLO

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

dpt = torch.hub.load('intel-isl/MiDaS', 'DPT_Large')
dpt.to(device)
dpt.eval()

transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
transform = transforms.dpt_transform

yolo = YOLO('model/yolov8l-face.pt')

def DistanceEstimation(relative_dis: float):
    return 235.61763103 - 7.45535009*relative_dis + 0.017384634*(relative_dis**2)
    
def FaceDetection(img):
    global yolo

    results = yolo.predict(img, conf=0.7)

    if len(results)>0:
        boxes = results[0].boxes.xyxy.tolist()
        pt1 = (int(boxes[0][0]), int(boxes[0][1]))
        pt2 = (int(boxes[0][2]), int(boxes[0][3]))

        boundingbox = (pt1, pt2)
    else:
        boundingbox = None 
    return boundingbox

def CalibrateData(depthmap, boundingbox):
    pt1, pt2 = boundingbox
    output = output[pt1[1]:pt2[1], pt1[0]:pt2[0]]

    pass

def Normalize(img):
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    normalized_gray_image = cv.normalize(gray_img, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
    img = cv.cvtColor(normalized_gray_image, cv.COLOR_GRAY2BGR)
    return img

def DepthMap(img):
    global dpt, transform, device

    img_batch = transform(img).to(device)
    
    with torch.no_grad():
        prediction = dpt(img_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    output = prediction.cpu().numpy()
    return output

def main():
    pass

if __name__ == '__main__':
    pass
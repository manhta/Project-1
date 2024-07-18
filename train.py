import torch
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import glob
import pandas as pd
from ultralytics import YOLO

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

dpt = torch.hub.load('intel-isl/MiDaS', 'DPT_Large')
dpt.to(device)
dpt.eval()

transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
transform = transforms.dpt_transform

yolo = YOLO('model/yolov8l-face.pt')
yolo.to(device)

def DistanceEstimation(relative_dis: float):
    return round(235.61763103 - 7.45535009*relative_dis + 0.017384634*(relative_dis**2), 2)
    
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
    depthmap = depthmap[pt1[1]:pt2[1], pt1[0]:pt2[0]]

    h,w = depthmap.shape
    output = depthmap[:int(0.5*h),:].reshape(-1,1)

    temp = []
    mean = np.mean(depthmap)
    std = np.std(depthmap)

    for items in depthmap:
        if (mean-3*std)<=items[0] and items[0]<=(mean+3*std):
            temp.append(items[0])
    temp = np.array(temp)
    mean_calibrated_data = np.mean(temp)

    return mean_calibrated_data

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
    folder_path = 'data/images/depth_train'
    files = glob.glob(folder_path + '/*.png')

    temp = []

    for filename in files:
        img = cv.imread(filename)
        normalized_img = Normalize(img)
        calibrated_data = CalibrateData(DepthMap(normalized_img), boundingbox=FaceDetection(normalized_img))

        print(filename[-8:], calibrated_data)
        temp.append([filename[-8:-6], calibrated_data])
    
    temp = pd.DataFrame(temp, columns=['Ground truth', 'Predicted depth'])
    pd.DataFrame.to_csv(temp, 'result2.csv', index=False)


if __name__ == '__main__':
    main()
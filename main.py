'''
    Distance estimation using pinhole camera model
'''

import cv2 as cv
import numpy as np
import os
import torch
import pandas as pd
from ultralytics import YOLO

torch.cuda.set_device(0)

yolo = YOLO('model/yolov8x.pt')

fx = 582                       # x-axis focal length in pixels
fy = 764                       # y-axis focal length in pixels  
m,n = 640, 480                 # Image resolution
P = (320, 240)                 # Principal point
u0, v0 = P

phi, omega, kappa = np.radians(-27.5), np.radians(0), np.radians(0)        # Rotation angles in radian

R = np.array([                                   # Rotation matrix
    [1,0,0], 
    [0, np.cos(phi), -np.sin(phi)], 
    [0, np.sin(phi), np.cos(phi)]
]).reshape(3,3)
R = np.around(R, decimals=3)

h = 1                                            # Camera height (m) 
C = np.array([0, h, 0]).reshape(-1,1)            # Camera in world coordinate 

t = -np.dot(R, C).reshape(-1,1)                  # Translation matrix
t = np.around(t, decimals=3)

R_inv = np.around(np.linalg.inv(R), decimals=3)  # Inverse rotation matrix

def InverseProjection(point_pixel: tuple):
    global R, R_inv, t
    global u0, v0
    global fx, fy
    
    u = point_pixel[0]
    v = point_pixel[1]
    v = 2*v0 - v

    r12 = R[1][2]
    r22 = R[2][2]
    ty = t[1][0]
    tz = t[2][0]

    z_c = round((ty*r22-tz*r12)/(((v-v0)*r22)/fy - r12),3)
    x_c = round(((u-u0)/fx) * z_c, 3)
    y_c = round(((v-v0)/fy) * z_c, 3)
    Xc = np.array([x_c, y_c, z_c]).reshape(-1,1)

    Xw = np.around(np.dot(R_inv, (Xc-t)),3)
    temp = Xw.reshape(1,-1)[0]
    return Xw

def DistanceEstimation(point_pixel: tuple):
    point = InverseProjection(point_pixel)

    x = point[0][0]
    z = point[2][0]
    
    distance = np.sqrt(x**2 + z**2)
    return round(distance,3)

def main():
    global yolo

    cam = cv.VideoCapture(0)

    while True:
        ret, frame = cam.read()

        if ret:
            frame = cv.flip(frame, 1)

            results = yolo.predict(frame, conf=0.7)

            if len(results)>0:
                boxes = results[0].boxes.xyxy.tolist()

                for box in boxes:
                    pt1 = (int(box[0]), int(box[1]))
                    pt2 = (int(box[2]), int(box[3]))

                    cv.rectangle(frame, pt1, pt2, (0,255,0), 1)

                    point = (int((pt1[0]+pt2[0])/2), pt2[1])
                    pred_dis = DistanceEstimation(point_pixel=point)
                    cv.putText(frame, f'Pred dis: {pred_dis} (m)', (pt1[0],pt1[1]-10), cv.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 1)

            cv.imshow('frame', frame)

            key = cv.waitKey(1)
            if key == ord('q'):
                break
        else:
            break

    cam.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
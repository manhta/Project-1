import numpy as np
from ultralytics import YOLO
import pandas as pd
import cv2 as cv
import os

def DistanceEstimation2(point: tuple):
    beta, alpha, gamma = np.radians(45), np.radians(17.5), np.radians(28.8)
    h = 1
    m,n = 640, 480
    u,v = point

    y = h*np.tan(beta +2*alpha*((n-1-v)/(n-1)))
    x = y*np.tan(gamma*((2*u-m+1)/(m-1)))

    print(x,y)
    return round(np.sqrt(x**2 +y**2),3)

model = YOLO('model/yolov8x.pt')

# cam = cv.VideoCapture(0)

# while True:
#     ret, frame = cam.read()

#     if ret:

#         cv.imshow('frame', frame)

#         key = cv.waitKey(1)
#         if key == ord('q'):
#             break
#     else:
#         break

# cam.release()
# cv.destroyAllWindows()

points = [(320,240), (409,248), (284,341), (58,390), (199,310), (521,302), (493,236), (485,439), (202,280), (128,254)] # sample points
res2 = []
gt = [1.956, 1.919, 1.432, 1.390, 1.650, 1.712, 2.073, 1.167, 1.756, 1.958]


for item in points:
    dis = DistanceEstimation2(item)
    res2.append(dis)

res_table = []

for i in range(len(res2)):
    res_table.append([res2[i], gt[i], round(abs(res2[i]-gt[i]), 3), round(abs(res2[i]-gt[i])/gt[i]*100,2)])

df = pd.DataFrame(res_table, columns=['Predict', 'Ground Truth', 'Error (m)', 'Error (%)'])
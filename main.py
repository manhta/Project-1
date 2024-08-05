import cv2 as cv
from ultralytics import YOLO
import numpy as np

K = np.load('data/result/calibrated_result/camera_intrinsic_matrix_3.npy')

fx = K[0][0]                        # x-axis focal length in pixels
fy = K[1][1]                        # y-axis focal length in pixels  
P = (320, 240)  # Principal point
u0, v0 = P

phi, omega, kappa = np.radians(27.5), np.radians(0), np.radians(0)        # Rotation angles in radian

R = np.array([                                  # Rotation matrix
    [1,0,0], 
    [0, np.cos(phi), -np.sin(phi)], 
    [0, np.sin(phi), np.cos(phi)]
]).reshape(3,3)
    
h = 1                                            # Camera height (m) 
d = 0                                            # Camera distance in world coordinates (m)
C = np.array([0, h, d]).reshape(-1,1)

# Translation matrix
t = -np.dot(R, C).astype(int).reshape(-1,1)

cam = cv.VideoCapture(0)
yolo = YOLO('model/yolov8x')

def InverseProjection(point_pixel: tuple):
    global K, R, t
    global u0, v0
    global fx, fy

    u = point_pixel[0]
    v = point_pixel[1]
    r21 = R[2][1]
    r22 = R[2][2]
    ty = t[1][0]
    tz = t[2][0]

    z_c = (ty*r22-tz*r21)/(((v-v0)*r22)/fy -r21)
    x_c = ((u-u0)/fx) * z_c
    y_c = ((v-v0)/fy) * z_c

    Xc = np.array([x_c, y_c, z_c]).reshape(-1,1)
    Xw = np.dot(np.linalg.inv(R), (Xc-t))

    return Xw

def DistanceEstimation(point_pixel: tuple):
    point = InverseProjection(point_pixel)

    x = point[0][0]
    z = point[2][0]

    distance = np.sqrt(x**2 + z**2)
    return round(distance,2)

# while True: 
#     ret, frame = cam.read()

#     if ret:
#         frame = cv.flip(frame, 1)

#         results = yolo.predict(frame, conf=0.5)

#         if len(results)>0:
#             boxes = results[0].boxes.xyxy.tolist()

#             for box in boxes:
#                 pt1 = (int(box[0]), int(box[1]))
#                 pt2 = (int(box[2]), int(box[3]))

#                 cv.rectangle(frame, pt1, pt2, (0,255,0), 1)

#                 u = int((pt1[0]+pt2[0])/2)
#                 v = pt2[1]
#                 p = (u,v)

#                 cv.putText(frame, f'Distance: {DistanceEstimation(p)} m', (pt1[0], pt1[1]-10), cv.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 1)
#                 cv.putText(frame, f'Point: {p}', (20,20), cv.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 1)
                
#         cv.imshow('frame', frame)

#         if cv.waitKey(1) == ord('q'):
#             break
#     else:
#         break

# cam.release()
# cv.destroyAllWindows()

print(R)
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

img_path = 'data/images/depth_train/40cm.png'
img = cv.imread(img_path)

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

print(1/(output))
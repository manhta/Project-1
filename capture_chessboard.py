import cv2
import os 

os.chdir('data/images/calibrated_images2')

cap = cv2.VideoCapture(0)

num = 0

while cap.isOpened():

    succes, img = cap.read()
    
    img = cv2.flip(img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corner = cv2.findChessboardCorners(gray, (7,9), None)
    k = cv2.waitKey(5)
    
    if ret: print(ret) 
    else: print('false')
    if k == ord('q'):
        break
    elif k == ord('s'): # wait for 's' key to save and exit
        cv2.imwrite('img' + str(num) + '.png', img)
        print("image saved!")
        num += 1

    cv2.imshow('Img',img)

# Release and destroy all windows before termination
cap.release()

cv2.destroyAllWindows()
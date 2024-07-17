import cv2 as cv
import os

os.chdir(os.chdir('data/depth_train'))

cam = cv.VideoCapture(0)
filename = ''

while True:
    ret, frame = cam.read()

    if ret:
        frame = cv.flip(frame, 1)

        cv.imshow('frame', frame)

        key = cv.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('w'):
            filename = str(input('Nhap ten file: '))
        elif key == ord('c'):
            if len(filename) == 0:
                print('Nhap ten file truoc')
            else:
                cv.imwrite(filename, frame)
    else:
        break

cam.release()
cv.destroyAllWindows()
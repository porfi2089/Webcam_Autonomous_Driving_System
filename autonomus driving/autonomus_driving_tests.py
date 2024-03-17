import cv2 as cv
import time
import numpy as np
from matplotlib import pyplot as plt
import json
import math


def load_cam_calib_data(filepath='camCalibrationData.json'):
    with open(filepath, 'r') as f:
        camCalibData = json.loads(f.read())

    global cameraMatrix
    global dist
    cameraMatrix = np.array(camCalibData['cameraMatrix'])
    dist = np.array(camCalibData['dist'])


def undistort_img(_img, _cameraMatrix, _dist):
    h,  w = _img.shape[:2]
    newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(_cameraMatrix, _dist, (w,h), 1, (w,h))

    # Undistort with Remapping
    mapx, mapy = cv.initUndistortRectifyMap(_cameraMatrix, _dist, None, newCameraMatrix, (w,h), 5)
    dst = cv.remap(_img, mapx, mapy, cv.INTER_LINEAR)

    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    return dst


load_cam_calib_data('../calibration/camCalibrationData.json')

start = time.time()

img = cv.imread('images/cam 2/WIN_20240315_14_44_22_Pro.jpg')

end = time.time()
totalTime = end - start
print("Time: ", totalTime)

img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
#img = undistort_img(img, cameraMatrix, dist)

end = time.time()
totalTime = end - start
print("Time: ", totalTime)

#img = cv.resize(img, (640, 480))
b = img.copy()
b[:, :, 2] = 0
b[:, :, 1] = 0

end = time.time()
totalTime = end - start
print("Time: ", totalTime)


gray = cv.cvtColor(b, cv.COLOR_RGB2GRAY)
#gray = cv.GaussianBlur(gray, (5, 5), 10)

threshhold1 = 200
threshhold2 = 120
mask1 = cv.bitwise_not(cv.threshold(gray, threshhold1, threshhold2, cv.THRESH_OTSU)[1])
mask = cv.bitwise_not(cv.adaptiveThreshold(gray, threshhold1, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 15, 8))

end = time.time()
totalTime = end - start
print("Time: ", totalTime)

mask = cv.bitwise_and(mask, mask1)
mask = cv.inRange(mask, 200, 255)

end = time.time()
totalTime = end - start
print("Time: ", totalTime)

can = cv.Canny(mask, 100, 125, apertureSize=7, L2gradient=True)
'''
lines = cv.HoughLines(
    can,
    1,
    np.pi/90,
    100,
    min_theta=0,
    max_theta=np.pi
)
'''

lines = cv.HoughLinesP(
    can,
    1,
    np.pi/90,
    100,
    minLineLength=50,
    maxLineGap=100
)

end = time.time()
totalTime = end - start
print("Time: ", totalTime)

lines_ = []
img_lines = img.copy()

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        lines_.append([x1, y1, x2, y2, (math.atan2(y2 - y1, x2 - x1) * 180 / np.pi)])

    for i in range(2):
        for line in lines_:
            x1, y1, x2, y2, angle = line
            print("evaluating", line)
            print()
            for line_ in lines_:
                x1_, y1_, x2_, y2_, angle_ = line_
                if line == line_:   # Skip the same line
                    continue
                if angle > 90:
                    angle = 180 - angle
                if 10 > abs(angle - angle_) > 0:
                    print("Removed: ", line_)
                    print("diff: ", abs(angle - angle_))
                    lines_.remove(line_)
                else:
                    print("Not Removed: ", line_)
                    print("diff: ", abs(angle - angle_))
            print()

    if 2 <= lines_.__len__() < 4:
        print("Lines: ", lines_)
        x1, y1, x2, y2, angle = lines_[0]
        x = [x1, x2]
        y = [y1, y2]
        x.sort()
        y.sort()
        x1, x2 = x
        y1, y2 = y
        slope = (y2 - y1) / (x2 - x1)
        b = y1 - slope * x1
        lines_[0][0] = int((img.shape[0] - b) / slope)
        lines_[0][1] = img.shape[0]
        lines_[0][2] = int((lines_[1][3] - b) / slope)
        lines_[0][3] = lines_[1][3]
        print("Line 0: ", lines_[0])
        x1, y1, x2, y2, angle = lines_[1]
        x = [x1, x2]
        x_ = [x1, x2]
        x.sort()
        if x != x_:
            x1, x2 = x_
            y = [y1, y2]
            y2, y1 = y
        slope = (y2 - y1) / (x2 - x1)
        b = y1 - slope * x1
        lines_[1][0] = int((img.shape[0] - b) / slope)
        lines_[1][1] = img.shape[0]
        print("Line 1: ", lines_[1])

        error_line = [int((lines_[0][0] + lines_[1][0]) / 2), img.shape[0], int((lines_[0][2] + lines_[1][2]) / 2), lines_[1][3], 0]
        x1, y1, x2, y2, null = error_line
        error_line[4] = (math.atan2(y2 - y1, x2 - x1) * 180 / np.pi)
        error = abs(error_line[4])-90
        print("Error Line: ", error_line)
        cv.line(img_lines, (x1, y1), (x2, y2), (100, 100, 255), 5)
        cv.putText(img_lines, " Error: "+str("{:.2f}".format(error)), (int((x2 + x1) / 2), int((y2 + y1) / 2)),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv.LINE_AA)
    for line in lines_:
        x1, y1, x2, y2, angle = line
        cv.line(img_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv.putText(img_lines, " Line "+str(lines_.index(line)), (int((x2+x1)/2), int((y2+y1)/2)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv.LINE_AA)
        cv.circle(img_lines, (x1, y1), 5, (255, 0, 0), -1)
        cv.putText(img_lines, str(x1)+", "+str(y1), (x1, y1), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv.LINE_AA)
        cv.circle(img_lines, (x2, y2), 5, (255, 255, 0), -1)
        cv.putText(img_lines, str(x2) + ", " + str(y2), (x2, y2), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv.LINE_AA)
        print(angle)


    print(lines.shape)
    print(lines_.__len__())

end = time.time()
totalTime = end - start
fps = 1 / totalTime
print("FPS: ", fps)


titles = ['Original Image', 'Canny', 'Mask', 'gray']
images = [img, can, mask, gray]

for i in range(4):
    plt.subplot(2, 2, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

cv.imshow("image", img_lines)
plt.show()

cv.waitKey(0)





import cv2 as cv
import time
import numpy as np
from matplotlib import pyplot as plt
import json
import math

# Load procedures for Json files
def load_config_data(filepath='autonomus_driving\config_data.json'):
    with open(filepath, 'r') as f:
        configData = json.loads(f.read())

    global lineReductions
    global camIndex
    global thresholds
    global frameSize
    global adaptiveSettings
    global horizontalLineThreshold

    lineReductions = configData['line_reductions']
    camIndex = configData['cam_index']
    thresholds = np.array(configData['thresholds'])
    frameSize = np.array(configData['frame_size'])
    adaptiveSettings = np.array(configData['adaptive_settings'])
    horizontalLineThreshold = configData["horizontal_line_threshold"][0]


def load_cam_calib_data(filepath='camCalibrationData.json'):
    with open(filepath, 'r') as f:
        camCalibData = json.loads(f.read())

    global cameraMatrix
    global dist
    cameraMatrix = np.array(camCalibData['cameraMatrix'])
    dist = np.array(camCalibData['dist'])


def load_unwrap_data(filepath='unwrap_data.json'):
    with open(filepath, 'r') as f:
        unwrapData = json.loads(f.read())

    global unwrap_cent
    unwrap_cent = np.array(unwrapData['centers'])


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


def load_files():
    load_config_data()
    load_cam_calib_data('calibration\camCalibrationData.json')
    load_unwrap_data(r'autonomus_driving\unwrap_data.json')


def get_frame(cap, img=None):
    try:
        ret, _img = cap.read()
    except:
        print("could not grab frame")
        return None
    if img is not None:
        if np.array_equal(_img, img):
            return None
            print("Same frame")
    return _img


def convert_frame(img, cameraMatrix, dist, matrix, frameSize):
    # img = undistort_img(img, cameraMatrix, dist)
    img = cv.warpPerspective(img, matrix, (frameSize[0], frameSize[1]))
    return img


def filter_RGB(img):
    r = img.copy()
    b = img.copy()
    g = img.copy()
    r[:, :, 0] = 0
    r[:, :, 1] = 0
    g[:, :, 0] = 0
    g[:, :, 2] = 0
    b[:, :, 2] = 0
    b[:, :, 1] = 0
    return r, g, b


def convert_to_gray(img):
    return cv.cvtColor(img, cv.COLOR_RGB2GRAY)


def apply_thresholds(gray, thresholds, adaptiveSettings):
    mask1 = cv.bitwise_not(cv.threshold(gray, thresholds[0], thresholds[1], cv.THRESH_OTSU)[1])
    mask = cv.bitwise_not(cv.adaptiveThreshold(gray, thresholds[0], cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, adaptiveSettings[0], adaptiveSettings[1]))
    mask = cv.bitwise_and(mask, mask1)
    mask = cv.inRange(mask, 200, 255)
    return mask


def apply_canny(mask):
    return cv.Canny(mask, 100, 125, apertureSize=7, L2gradient=True)


def get_lines(can):
    return cv.HoughLinesP(
        can,
        1,
        np.pi/90,
        100,
        minLineLength=50,
        maxLineGap=100
    )


def calculate_slope_and_intercept(x1, y1, x2, y2):
    x = np.sort([x1, x2])
    y = np.sort([y1, y2])
    x1, x2 = x
    y1, y2 = y
    x2 = x2 if x1 != x2 else x2+0.5
    slope = (y2 - y1) / (x2 - x1)
    slope = slope if slope != 0 else 0.05
    b = y1 - slope * x1
    return slope, b


def process_2lines(lines_, img):
    x1, y1, x2, y2, angle = lines_[0]
    x = np.sort([x1, x2])
    y = np.sort([y1, y2])
    x1, x2 = x
    y1, y2 = y
    x2 = x2 if x1 != x2 else x2+0.5
    slope = (y2 - y1) / (x2 - x1)
    slope = slope if slope != 0 else 0.05
    b = y1 - slope * x1
    lines_[0][0] = int((img.shape[0] - b) / slope)
    lines_[0][1] = img.shape[0]
    lines_[0][2] = int((lines_[1][3] - b) / slope)
    lines_[0][3] = lines_[1][3]

    x1, y1, x2, y2, angle = lines_[1]
    x = [x1, x2]
    x_ = [x1, x2]
    x.sort()
    if x != x_:
        x1, x2 = x_
        y = [y1, y2]
        y2, y1 = y
    if x1 == x2:
        x2 = x2+0.5
    slope = (y2 - y1) / (x2 - x1)
    if slope == 0:
        slope = 0.05
    b = y1 - slope * x1
    lines_[1][0] = int((img.shape[0] - b) / slope)
    lines_[1][1] = img.shape[0]
    return lines_


def join_list_of_subs(list1, list2):
    list3 = []
    for i in range(list1.__len__()):
        list3.append(list1[i])
    for i in range(list2.__len__()):
        list3.append(list2[i])
    return list3


def process_errorLine(error_line, text="Error: "):
    x1, y1, x2, y2, null = error_line
    error_line[4] = (math.atan2(y2 - y1, x2 - x1) * 180 / np.pi)
    error = abs(error_line[4])-90
    cv.line(img_lines, (x1, y1), (x2, y2), (100, 100, 255), 5)
    cv.putText(img_lines, str(text)+str("{:.2f}".format(error)), (int((x2 + x1) / 2), int((y2 + y1) / 2)),
                cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv.LINE_AA)
    return error


def draw_line(img_lines, line):
    x1, y1, x2, y2, angle = line
    cv.line(img_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv.putText(img_lines, " Line "+str(lines_.index(line)), (int((x2+x1)/2), int((y2+y1)/2)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv.LINE_AA)
    cv.circle(img_lines, (x1, y1), 5, (255, 0, 0), -1)
    cv.putText(img_lines, str(x1)+", "+str(y1), (x1, y1), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv.LINE_AA)
    cv.circle(img_lines, (x2, y2), 5, (255, 255, 0), -1)
    cv.putText(img_lines, str(x2) + ", " + str(y2), (x2, y2), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv.LINE_AA)


def reduce_lines(lineReductions, lines_):
    for i in range(lineReductions):
            for line in lines_:
                x1, y1, x2, y2, angle = line
                # print("evaluating", line)
                # print()
                if angle > 90:
                    angle = -180 + angle
                    lines_[lines_.index(line)][4] = angle
                elif angle < -90:
                    angle = 180 + angle
                    lines_[lines_.index(line)][4] = angle
                for line_ in lines_:
                    x1_, y1_, x2_, y2_, angle_ = line_
                    if line == line_:   # Skip the same line
                        continue
                    if 10 > abs(angle - angle_) > 0:
                        # print("Removed: ", line_)
                        # print("diff: ", abs(angle - angle_))
                        lines_.remove(line_)
                    else:
                        # print("Not Removed: ", line_)
                        # print("diff: ", abs(angle - angle_))
                        pass

# initialize navCam recording
def init_rec(camIndex, frameSize, unwrap_cent):
    cap = cv.VideoCapture(camIndex)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, frameSize[0])
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, frameSize[1])

    # generate transformation matrix
    pts2 = [[0, 0], [0, frameSize[1]], [frameSize[0], 0], [frameSize[0], frameSize[1]]]
    matrix = cv.getPerspectiveTransform(np.float32(unwrap_cent), np.float32(pts2))
    return cap, matrix


def check_side_of_point(point, line):
    x1, y1, x2, y2, angle = line
    if x1 == x2:
        x2 = x2+0.5
    slope, intercept = calculate_slope_and_intercept(x1, y1, x2, y2)
    y = slope * point[0] + intercept
    if point[1] < y:
        return "above"
    elif point[1] > y:
        return "below"
    else:
        return "on"
    

#init variables
error = 0
#flags
turnCentanty = 0
turn = False
turnRight = False
turnLeft = False

load_files()
cap, matrix = init_rec(camIndex, frameSize, unwrap_cent)

while cap.isOpened():
    try:
        start
    except NameError:
        start = time.time()
    try:
        img
    except NameError:
        img = None

    img = get_frame(cap, img)
    img = cv.imread(r"autonomus_driving\images\cam 2\WIN_20240315_14_44_07_Pro.jpg")
    if img is None:
        continue
    img = convert_frame(img, cameraMatrix, dist, matrix, frameSize)
    r, g, b = filter_RGB(img)
    gray = convert_to_gray(b)
    mask = apply_thresholds(gray, thresholds, adaptiveSettings)
    can = apply_canny(mask)
    lines = get_lines(can)
    lines_ = []
    img_lines = img.copy()

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            lines_.append([x1, y1, x2, y2, -90+(math.atan2(y2 - y1, x2 - x1) * 180 / np.pi)])        
        
        reduce_lines(lineReductions, lines_)        

        vert_lines = []
        hori_lines = []

        for line in lines_:
            x1, y1, x2, y2, angle = line
            if -45 < angle < 45:
                vert_lines.append(line)
            else:
                hori_lines.append(line)

        lines_ = join_list_of_subs(vert_lines, hori_lines)
        
        if 2 <= vert_lines.__len__():

            process_2lines(lines_, img)

            error_line = [int((lines_[0][0] + lines_[1][0]) / 2), img.shape[0], int((lines_[0][2] + lines_[1][2]) / 2), lines_[1][3], 0]
            
            error = process_errorLine(error_line)

        elif vert_lines.__len__() == 1:

            error_line = lines_[0]

            
            error = process_errorLine(error_line)

        if hori_lines.__len__() == 1:
            hori_lines = hori_lines[0]
            x1, y1, x2, y2, angle = hori_lines
            hori_line = hori_lines
                
        if hori_lines.__len__() == 2:
            hori_lines = process_2lines(hori_lines, img)
            hori_line = [int((hori_lines[0][0] + hori_lines[1][0]) / 2), img.shape[0], int((hori_lines[0][2] + hori_lines[1][2]) / 2), hori_lines[1][3], 0]

        if hori_line == 1:
            x1, y1, x2, y2, angle = hori_line
            cv.line(img_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv.putText(img_lines, " Line 1", (int((x2+x1)/2), int((y2+y1)/2)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv.LINE_AA)
            x = (x1 + x2) / 2
            y = (y1 + y2) / 2
            if turn == False:
                if y > horizontalLineThreshold:
                    if turnCentanty < 5:
                        turnCentanty += 24
                    else:
                        turn = True
                        turnCentanty = 0
                elif y < horizontalLineThreshold:
                    if turnCentanty > 0:
                        turnCentanty -= 1
            else:
                if not turnRight and not turnLeft:
                    side = check_side_of_point([x, y], error_line)
                    if side == "above":
                        turnLeft = True
                    elif side == "below":
                        turnRight = True

        for line in lines_:
            draw_line(img_lines, line)
            

    end = time.time()
    totalTime = end - start
    start = time.time()
    fps = 1 / totalTime
    totalTime_Ms = (totalTime * 1000)
    print("Time: ", str("{:.2f}".format(totalTime_Ms)) + "ms")
    print("FPS: ", fps)
    can = cv.cvtColor(can, cv.COLOR_GRAY2BGR)
    img_lines = cv.addWeighted(img_lines, 1, can, 0.8, 0)
    cv.putText(img_lines, f'error: {"{:.4f}".format(error)}°', (20, img.shape[0] - 70), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv.putText(img_lines, f'FPS: {int(fps)}', (20, 70), cv.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
    cv.imshow("image", img_lines)
    key = cv.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    time.sleep(0.1)


plt.show()

cv.waitKey(0)


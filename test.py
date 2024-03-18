import cv2 as cv
import time
import numpy as np

img = cv.imread('autonomus driving/images/cam 2/WIN_20240315_14_44_22_Pro.jpg')

pts1 = [(125, 100), (0, 380), (450, 100), (600, 380)]
pts2 = [[0, 0], [0, 480], [640, 0], [640, 480]]

cv.circle(img, pts1[0], 5, (0, 0, 255), -1)
cv.putText(img, '1 '+str(pts1[0]), pts1[0], cv.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 2, cv.LINE_AA)
cv.circle(img, pts1[1], 5, (0, 0, 255), -1)
cv.putText(img, '2 '+str(pts1[1]), pts1[1], cv.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 2, cv.LINE_AA)
cv.circle(img, pts1[2], 5, (0, 0, 255), -1)
cv.putText(img, '3 '+str(pts1[2]), pts1[2], cv.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 2, cv.LINE_AA)
cv.circle(img, pts1[3], 5, (0, 0, 255), -1)
cv.putText(img, '4 '+str(pts1[3]), pts1[3], cv.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 2, cv.LINE_AA)

matrix = cv.getPerspectiveTransform(np.float32(pts1), np.float32(pts2))
result = cv.warpPerspective(img, matrix, (640, 480))

cv.imshow('result', result)
cv.imshow('img', img)
cv.waitKey(0)
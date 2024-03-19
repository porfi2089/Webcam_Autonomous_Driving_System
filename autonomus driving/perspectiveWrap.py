import cv2 as cv
import time
import numpy as np
from cv2 import aruco

marker_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_250)

param_markers = aruco.DetectorParameters()

cap = cv.VideoCapture(1)

Des_marker_IDs = [[14], [11], [15], [12]]
while True:
    start = time.time()
    ret, frame = cap.read()
    if not ret:
        break
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    marker_corners, marker_IDs, reject = aruco.detectMarkers(
        gray_frame, marker_dict, parameters=param_markers
    )
    centers = []
    if marker_corners:
        for ids, corners in zip(marker_IDs, marker_corners):
            cv.polylines(
                frame, [corners.astype(np.int32)], True, (0, 255, 255), 4, cv.LINE_AA
            )
            center = np.mean(corners, axis=1)
            centers.append(center)
            corners = corners.reshape(4, 2)
            corners = corners.astype(int)
            top_right = corners[0].ravel()
            top_left = corners[1].ravel()
            bottom_right = corners[2].ravel()
            bottom_left = corners[3].ravel()
            cv.putText(
                frame,
                f"id: {ids[0]}",
                top_right,
                cv.FONT_HERSHEY_PLAIN,
                1.3,
                (200, 100, 0),
                2,
                cv.LINE_AA,
            )

        print("Marker IDs: ", marker_IDs)
        if marker_IDs.size == 4:
            img = frame.copy()
            marker_indexs = []
            true_centers = []
            try:
                for did in Des_marker_IDs:
                    index = marker_IDs.tolist().index(did)
                    marker_indexs.append(index)
                    true_centers.append(centers[index])
                    print("Index: ", index)
            except:
                pass

            print(true_centers)
            pts1 = [(corners[3][0], corners[3][1]), (corners[2][0], corners[2][1]), (corners[0][0], corners[0][1]), (corners[1][0], corners[1][1])]
            pts2 = [[0, 0], [0, 480], [640, 0], [640, 480]]
            cv.circle(img, pts1[0], 5, (0, 0, 255), -1)
            cv.putText(img, '1 ' + str(pts1[0]), pts1[0], cv.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 2, cv.LINE_AA)
            cv.circle(img, pts1[1], 5, (0, 0, 255), -1)
            cv.putText(img, '2 ' + str(pts1[1]), pts1[1], cv.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 2, cv.LINE_AA)
            cv.circle(img, pts1[2], 5, (0, 0, 255), -1)
            cv.putText(img, '3 ' + str(pts1[2]), pts1[2], cv.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 2, cv.LINE_AA)
            cv.circle(img, pts1[3], 5, (0, 0, 255), -1)
            cv.putText(img, '4 ' + str(pts1[3]), pts1[3], cv.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 2, cv.LINE_AA)
            matrix = cv.getPerspectiveTransform(np.float32(pts1), np.float32(pts2))
            frame = cv.warpPerspective(frame, matrix, (640, 480))
            cv.imshow('result', img)

    end = time.time()
    totalTime = end - start
    fps = 1 / totalTime
    print("FPS: ", fps)
    cv.putText(frame, f'FPS: {int(fps)}', (20, 70), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
    cv.imshow("frame", frame)
    key = cv.waitKey(1)
    if key == ord("q"):
        break
cap.release()
cv.destroyAllWindows()


'''cv.circle(img, pts1[0], 5, (0, 0, 255), -1)
cv.putText(img, '1 '+str(pts1[0]), pts1[0], cv.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 2, cv.LINE_AA)
cv.circle(img, pts1[1], 5, (0, 0, 255), -1)
cv.putText(img, '2 '+str(pts1[1]), pts1[1], cv.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 2, cv.LINE_AA)
cv.circle(img, pts1[2], 5, (0, 0, 255), -1)
cv.putText(img, '3 '+str(pts1[2]), pts1[2], cv.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 2, cv.LINE_AA)
cv.circle(img, pts1[3], 5, (0, 0, 255), -1)
cv.putText(img, '4 '+str(pts1[3]), pts1[3], cv.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 2, cv.LINE_AA)'''

matrix = cv.getPerspectiveTransform(np.float32(pts1), np.float32(pts2))
result = cv.warpPerspective(img, matrix, (640, 480))

cv.imshow('result', result)
cv.imshow('img', img)
cv.waitKey(0)
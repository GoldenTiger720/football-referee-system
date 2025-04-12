from ultralytics import YOLO
import cv2
import numpy as np
import time

model = YOLO("yolov8x-seg.engine")
img_orig= cv2.imread("data/stitched_image.png")

for i in range(1):
    start_time = time.time()

    results = model(img_orig)


    img = np.copy(results[0].orig_img)
    b_mask = np.zeros(img.shape[:2], np.uint8)
    print(len(results[0].masks.xy))
    contour = results[0].masks.xy[2].astype(np.int32).reshape(-1, 1, 2)
    cv2.drawContours(b_mask, [contour], -1, (255, 255, 255), cv2.FILLED)

    mask3ch = cv2.cvtColor(b_mask, cv2.COLOR_GRAY2BGR)
    isolated = cv2.bitwise_and(mask3ch, img)

    mask3ch = cv2.cvtColor(b_mask, cv2.COLOR_GRAY2BGR)
    isolated = cv2.bitwise_and(mask3ch, img)

    end_time = time.time()

    elapsed_ms = (end_time-start_time)*1000
    print("Process time:", elapsed_ms)

cv2.imshow("X",isolated)
cv2.waitKey(1000)

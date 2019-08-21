#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
from facedetection import faceDetection


detector = faceDetection()

image = cv2.imread("E:/humAIn/humAIn_faceless/humAIn_FaceLess/images_TCS/supermarket-56.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
result = detector.detect_faces(image)
print(result)

# Result is an array with all the bounding boxes detected. We know that for 'ivan.jpg' there is only one.
for detection in result:
    bounding_box = detection['box']

    cv2.rectangle(image,
              (bounding_box[0], bounding_box[1]),
              (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
              (0,155,255), 2)


cropped = image[bounding_box[1]: bounding_box[1] + bounding_box[3], bounding_box[0]:bounding_box[0]+bounding_box[2]]

image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
cropped = cv2.cvtColor(cropped, cv2.xCOLOR_RGB2BGR)

cv2.imshow("frame", image)
cv2.imshow("croppedframe", cropped)
cv2.waitKey()


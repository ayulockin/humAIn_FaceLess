#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
from facedetection import faceDetection
from genderagerace import GenderRaceAge


detector = faceDetection()

image = cv2.imread("E:/humAIn/humAIn_faceless/humAIn_FaceLess/images_TCS/supermarket-56.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
result = detector.detect_faces(image)
print(result)

# image_bbox = detector.drawBoundingBox(image, result)


gra_detector = GenderRaceAge()
cropped_faces = detector.getCropedImages(image, result)
output = gra_detector.predict(cropped_faces)

print(output)


# for face_id , face in cropped_faces.items():
#   cv2.imshow(face_id, face)

# cv2.imshow("frame", image_bbox)

# cv2.waitKey()
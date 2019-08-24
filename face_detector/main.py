#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
from facedetection import faceDetection
from identifyface import IdentifyFace

## Prepare Detectors 
face_detector = faceDetection()
identity_detector = IdentifyFace()

## Load image
image = cv2.imread("E:/humAIn/humAIn_faceless/humAIn_FaceLess/images_TCS/supermarket-56.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

## Detect Face
result = face_detector.detect_faces(image)
print(result)
## Get cropped Faces
cropped_faces = face_detector.getCropedImages(image, result)

# image_bbox = face_detector.drawBoundingBox(image, result)

## Detect Identity
output = identity_detector.predict(cropped_faces)

print(output)


# for face_id , face in cropped_faces.items():
#   cv2.imshow(face_id, face)

# cv2.imshow("frame", image_bbox)

# cv2.waitKey()
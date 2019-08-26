#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import json
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=False,
	help="path to test image")
ap.add_argument("-b", "--bbox", required=False,
	help="draw predicted bounding box")
args = vars(ap.parse_args())

from face_detector.facedetection import faceDetection
from face_detector.identifyface import IdentifyFace

# Prepare Detectors 
face_detector = faceDetection()
identity_detector = IdentifyFace()

image_path = "E:/humAIn/humAIn_faceless/humAIn_FaceLess/images/1SZ5m0pmDcS6RgpM8FPNCuw.png"
# image_path = args['image']

## Load image
image = cv2.imread(image_path)

# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Detect Face
result = face_detector.detect_faces(image)

## Get cropped Faces
cropped_faces = face_detector.getCropedImages(image, result)

## Detect Identity
output = identity_detector.predict(cropped_faces)

print(output)

if args['bbox']:
	image = face_detector.drawBoundingBox(image, result)
	cv2.imshow("frame", image)
	cv2.waitKey()
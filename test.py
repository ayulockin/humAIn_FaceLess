import cv2
import os
import json
import argparse

from skimage.io import imread

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to test image")
ap.add_argument("-b", "--bbox", required=False,
	help="draw predicted bounding box")
args = vars(ap.parse_args())

from face_detector.facedetection import faceDetection
from face_detector.identifyface import IdentifyFace

# Prepare Detectors 
face_detector = faceDetection()
identity_detector = IdentifyFace('weights')


# Image path
image_path = args['image']

## Load image
image = cv2.imread(image_path)

# Detect Face
result = face_detector.detect_faces(image)
print(result)

# Get cropped Faces
cropped_faces = face_detector.getCropedImages(image, result)

print(len(cropped_faces))

for img_id, img in cropped_faces.items():
	print(img.shape)

for  img_id, cf in cropped_faces.items():
	
	cv2.imshow(img_id, cf)
	cv2.waitKey()

# ## Detect Identity
output = identity_detector.predict(cropped_faces)

print(output)

if args['bbox']:
	image = face_detector.drawBoundingBox(image, result)
	cv2.imshow("frame", image)
	cv2.waitKey()
import cv2
import os
import json
import base64

from skimage.io import imread

from face_detector.facedetection import faceDetection
from face_detector.identifyface import IdentifyFace

# Prepare Detectors 
face_detector = faceDetection()
identity_detector = IdentifyFace('weights')


def working(imagefile):

	image = imread(imagefile)
	
	# Detect Face
	result = face_detector.detect_faces(image)
	print(result)

	# Get cropped Faces
	cropped_faces = face_detector.getCropedImages(image, result)

	print(len(cropped_faces))

	for img_id, img in cropped_faces.items():
		print(img.shape)
	# ## Detect Identity
	# image = face_detector.drawBoundingBox(image, result)

	output = identity_detector.predict(cropped_faces)

	image = face_detector.drawBoundingBox(image, result)
	
	return image
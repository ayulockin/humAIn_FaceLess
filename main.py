import cv2
import os
import json
import argparse

from skimage.io import imread

from face_detector.facedetection import faceDetection
from face_detector.identifyface import IdentifyFace

# Prepare Detectors 
face_detector = faceDetection()
identity_detector = IdentifyFace()


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

	# image = face_detector.drawBoundingBox(image, result)
	# ## Detect Identity
	output = identity_detector.predict(cropped_faces)

	return (output)
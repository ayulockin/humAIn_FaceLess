#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import os
import json
from face_detector.facedetection import faceDetection
from face_detector.identifyface import IdentifyFace

def getAnnotationImage(image_path):
	image_name = image_path[image_path.rfind("/")+1:]
	with open('datasets/annotations.json', encoding='utf8') as json_file:
		data_dict = json.loads(json_file.read())
	
	return data_dict['data'][image_name]

def computeBBox(annotations):
	faces_id = {}
	face_count = 0

	for annotation in annotations:
		faces_id['face_{}'.format(face_count)] = {}
		true_label = annotation['label']

		img_W = annotation['imageWidth']
		img_H = annotation['imageHeight']


		points = annotation['points']
		box = []
		for point in points:
			for key, val in point.items():
				if key=='x':
					box.append(int(round(val*img_W)))
				else:
					box.append(int(round(val*img_H)))

		faces_id['face_{}'.format(face_count)]['box'] = box
		faces_id['face_{}'.format(face_count)]['label'] = true_label
		face_count+=1

	return faces_id

def drawTrueBBox(image, faces_id):

	for face_id in faces_id:
		bounding_box = faces_id[face_id]['box']
		cv2.rectangle(image,
	                      (bounding_box[0], bounding_box[1]),
	                      (bounding_box[2], bounding_box[3]),
	                      (0,0,255), 2)

	return image

# Prepare Detectors 
face_detector = faceDetection()
identity_detector = IdentifyFace()

image_path = "E:/humAIn/humAIn_faceless/humAIn_FaceLess/images/ALDI_NewToALDI_ShoppingAtALDI_InformationPage_316x284_Desktop_2.jpg.jpeg"

## Load image
image = cv2.imread(image_path)

# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Detect Face
result = face_detector.detect_faces(image)
print(result)
## Get cropped Faces
cropped_faces = face_detector.getCropedImages(image, result)

## Detect Identity
output = identity_detector.predict(cropped_faces)

print(output)

trueannotations = getAnnotationImage(image_path)
# print(trueannotations)
faces_id = computeBBox(trueannotations)
print(faces_id)

image = face_detector.drawBoundingBox(image, result)
image = drawTrueBBox(image, faces_id)

cv2.imshow("frame", image)

for face_id, img in cropped_faces.items():
	cv2.imshow(face_id, img)



cv2.waitKey()
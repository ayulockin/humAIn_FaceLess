import numpy as np 
import pandas as pd
import os

import cv2

import tensorflow as tf
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

import keras.backend as K
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense, Flatten, BatchNormalization, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.models import model_from_yaml

from keras.models import load_model

class IdentifyFace():
	def __init__(self, weight_path):

		self.weight_path = weight_path

		self._gen_eth_weight_path = os.path.join(weight_path+'/gender_ethnicity.h5')
		self._emotion_weight_path = os.path.join(weight_path+'/emotion.h5')
		self._age_weight_path = os.path.join(weight_path+'/age.h5')


		self.__gen_eth_model = None
		self.__emotion_model = None
		self.__age_model = None

		self.__graph1 = None
		self.__graph2 = None
		self.__graph3 = None
		self.__session1 = None
		self.__session2 = None
		self.__session3 = None

		print("[INFO] Preparing Model and loading weights...")
		self.gen_eth_buildModel()
		self.emotion_buildModel()
		self.age_buildModel()
		print("[INFO] Done")


	# Model for Gender and Ethnicity
	def gen_eth_buildModel(self):
		self.__graph1 = tf.Graph()
		with self.__graph1.as_default():
			self.__session1 = tf.Session()
			with self.__session1.as_default():
				gen_ethn_file = open(os.path.join(self.weight_path+'/model_gender_ethnicity.yaml'), 'r')
				loaded_model_yaml = gen_ethn_file.read()
				gen_ethn_file.close()
				loaded_model = model_from_yaml(loaded_model_yaml)
				# load weights into new model
				loaded_model.load_weights(self._gen_eth_weight_path)
				self.__gen_eth_model = loaded_model


	# Model for emotion
	def emotion_buildModel(self):
		self.__graph2 = tf.Graph()
		with self.__graph2.as_default():
			self.__session2 = tf.Session()
			with self.__session2.as_default():	
				emotion_file = open(os.path.join(self.weight_path+'/model_emotion.yaml'), 'r')
				emotion_model_yaml = emotion_file.read()
				emotion_file.close()
				emotion_model = model_from_yaml(emotion_model_yaml)
				# load weights into new model
				emotion_model.load_weights(self._emotion_weight_path)
				self.__emotion_model = emotion_model

	# Model for age
	def age_buildModel(self):
		self.__graph3 = tf.Graph()
		with self.__graph3.as_default():
			self.__session3 = tf.Session()
			with self.__session3.as_default():
				age_file = open(os.path.join(self.weight_path+'/model_age.yaml'), 'r')
				age_model_yaml = age_file.read()
				age_file.close()
				age_model = model_from_yaml(age_model_yaml)
				# load weights into new model
				age_model.load_weights(self._age_weight_path)
				self.__age_model = age_model


	# Prediction from all three classifiers. 
	def predict(self, image_dict):
		print("[INFO] Predicting Gender, Ethnicity, Age and Emotion...")
		output = {}
		for face_id, image in image_dict.items():

			gen_eth_img = cv2.resize(image, self.__gen_eth_model.layers[0].output_shape[1:3][::-1])
			emotion_img = cv2.resize(image, self.__emotion_model.layers[0].output_shape[1:3][::-1])
			age_img = cv2.resize(image, self.__age_model.layers[0].output_shape[1:3][::-1])

			gen_eth_img = gen_eth_img.reshape((1,)+gen_eth_img.shape)
			age_img = age_img.reshape((1,)+age_img.shape)
			emotion_img = emotion_img.reshape((1,)+emotion_img.shape)


			with self.__graph1.as_default():
				with self.__session1.as_default():
					gender_ethnicity = self.__gen_eth_model.predict(gen_eth_img)
					# print("[CHECKING] Gender :", gender_ethnicity)
					# print("[CHECKING] Gender:", np.argmax(gender_ethnicity))
					gender, ethnicity = self.__decodeGenderEthnicity(np.argmax(gender_ethnicity))
					

			with self.__graph2.as_default():
				with self.__session2.as_default():
					emotion = self.__emotion_model.predict(emotion_img)
					emotion = self.__decodeEmotion(np.argmax(emotion))

			with self.__graph3.as_default():
				with self.__session3.as_default():
					age_prediction = self.__age_model.predict(age_img)
					# print("[CHECKING] Age:", age_prediction)
					# print("[CHECKING] Age:", np.argmax(age_prediction))
					age = self.__degroupAge(np.argmax(age_prediction))

			output[face_id] = [gender, ethnicity, age, emotion]

		print("[INFO] Done")
		return output

	def __decodeGenderEthnicity(self, ged_eth):
		if ged_eth==0:
			return 'G_Male', 'E_White'
		elif ged_eth==1:
			return 'G_Male', 'E_Hispanic'
		elif ged_eth==2:
			return 'G_Male', 'E_Asian'
		elif ged_eth==3:
			return 'G_Male', 'E_Black'
		elif ged_eth==4:
			return 'G_Male', 'E_Arab'
		elif ged_eth==5:
			return 'G_Male', 'E_Indian'
		elif ged_eth==6:
			return 'G_Female', 'E_White'
		elif ged_eth==7:
			return 'G_Female', 'E_Hispanic'
		elif ged_eth==8:
			return 'G_Female', 'E_Asian'
		elif ged_eth==9:
			return 'G_Female', 'E_Black'
		elif ged_eth==10:
			return 'G_Female', 'E_Arab'
		elif ged_eth==11:
			return 'G_Female', 'E_Indian'

	# Label Age's prediction
	def __degroupAge(self, age):
	    if age==0:
	        return 'Age_below20'
	    if age==1:
	        return 'Age_20_30'
	    if age==2:
	        return 'Age_30_40'
	    if age==3:
	        return 'Age_40_50'
	    if age==4:
	        return 'Age_above_50'

	# Label Emotion's prediction
	def __decodeEmotion(self, emotion):
	    if emotion==0:
	        return 'Emotion_Happy'
	    if emotion==1:
	        return 'Emotion_Neutral'
	    if emotion==2:
	        return 'Emotion_Sad'
	    if emotion==3:
	        return 'Emotion_Angry'

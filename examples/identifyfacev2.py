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
	def __init__(self, weight_path=['face_detector/weights/gender-ethnicity.hdf5', 
									'face_detector/weights/emotion.hdf5',
									'face_detector/weights/age.hdf5']):

		self._ge_weight_path = weight_path[0]
		self._e_weight_path = weight_path[1]
		self._a_weight_path = weight_path[2]

		self.__ge_model = None
		self.__e_model = None
		self.__a_model = None

		self.__graph1 = None
		self.__graph2 = None
		self.__graph3 = None
		self.__session1 = None
		self.__session2 = None
		self.__session3 = None

		print("[INFO] Preparing Model and loading weights...")
		self.ge_buildModel()
		self.e_buildModel()
		self.a_buildModel()
		print("[INFO] Done")


	# Model for Gender and Ethnicity
	def ge_buildModel(self):
		self.__graph1 = tf.Graph()
		with self.__graph1.as_default():
			self.__session1 = tf.Session()
			with self.__session1.as_default():
				yaml_file = open('face_detector/weights/model-gender-ethnicity.yaml', 'r')
				loaded_model_yaml = yaml_file.read()
				yaml_file.close()
				loaded_model = model_from_yaml(loaded_model_yaml)
				# load weights into new model
				loaded_model.load_weights(self._ge_weight_path)
				self.__ge_model = loaded_model

	# Model for age
	def a_buildModel(self):
		self.__graph3 = tf.Graph()
		with self.__graph3.as_default():
			self.__session3 = tf.Session()
			with self.__session3.as_default():
				age_file = open('face_detector/weights/model-age.yaml', 'r')
				age_model_yaml = age_file.read()
				age_file.close()
				age_model = model_from_yaml(age_model_yaml)
				# load weights into new model
				age_model.load_weights(self._a_weight_path)
				self.__a_model = age_model

	# Model for emotion
	def e_buildModel(self):
		self.__graph2 = tf.Graph()
		with self.__graph2.as_default():
			self.__session2 = tf.Session()
			with self.__session2.as_default():	
				inputs = Input(shape=(48,48,1))
				emotion = self.__emotion_classification(inputs)
				e_model = Model(inputs=inputs, outputs=emotion)
				e_model = load_model(self._e_weight_path)
				self.__e_model = e_model


	# Prediction from all three classifiers. 
	def predict(self, image_dict):
		print("[INFO] Predicting Gender, Ethnicity, Age and Emotion...")
		output = {}
		for face_id, image in image_dict.items():

			gra_image = cv2.resize(image, (200,200))

			e_image = cv2.resize(image, (48,48))
			e_image = cv2.cv2.cvtColor(e_image, cv2.COLOR_RGB2GRAY)
			
			gra_image = gra_image.reshape(((1,)+gra_image.shape))
			e_image = e_image.reshape((1,48,48,1))

			with self.__graph1.as_default():
				with self.__session1.as_default():
					gender_ethnicity = self.__ge_model.predict(gra_image)
					# print("[CHECKING] Gender :", gender_ethnicity)
					# print("[CHECKING] Gender:", np.argmax(gender_ethnicity))
					
					gender, ethnicity = self.__decodeGenderEthnicity(np.argmax(gender_ethnicity))
					

			with self.__graph2.as_default():
				with self.__session2.as_default():
					emotion = self.__e_model.predict(e_image)

					emotion = self.__decodeEmotion(np.argmax(emotion))

			with self.__graph3.as_default():
				with self.__session3.as_default():
					age_prediction = self.__a_model.predict(gra_image)
					# print("[CHECKING] Age:", age_prediction)
					# print("[CHECKING] Age:", np.argmax(age_prediction))
					age = self.__degroupAge(np.argmax(age_prediction))

			output[face_id] = [gender, ethnicity, age, emotion]

		print("[INFO] Done")
		return output

	# Decode prediction for Gender and Ethnicity
	def __decodeGenderEthnicity(self, gender_ethnicity):
		if gender_ethnicity==0:
		    return self.__decodeGender(0), self.__decodeEthnicity(0)
		elif gender_ethnicity==1:
			return self.__decodeGender(0), self.__decodeEthnicity(1)
		elif gender_ethnicity==2:
			return self.__decodeGender(0), self.__decodeEthnicity(2)
		elif gender_ethnicity==3:
			return self.__decodeGender(0), self.__decodeEthnicity(3)
		elif gender_ethnicity==4:
			return self.__decodeGender(0), self.__decodeEthnicity(4)
		elif gender_ethnicity==5:
			return self.__decodeGender(1), self.__decodeEthnicity(0)
		elif gender_ethnicity==6:
			return self.__decodeGender(1), self.__decodeEthnicity(1)
		elif gender_ethnicity==7:
			return self.__decodeGender(1), self.__decodeEthnicity(2)
		elif gender_ethnicity==8:
			return self.__decodeGender(1), self.__decodeEthnicity(3)
		else:
			return self.__decodeGender(1), self.__decodeEthnicity(4)

	# Label Gender's prediction
	def __decodeGender(self, gender):
		if gender==0:
			return 'male'
		else:
			return 'female'

	# Label Ethnicity's prediction
	def __decodeEthnicity(self, ethnicity):
		if ethnicity==0:
			return 'white'
		elif ethnicity==1:
			return 'black'
		elif ethnicity==2:
			return 'asian'
		elif ethnicity==3:
			return 'indian'
		elif ethnicity==4:
			return 'haspanic or arabic'

	# Label Age's prediction
	def __degroupAge(self, age):
		# Age_below20
		if age==0 or age==1:
			return 'age_below20'

		# Age_20_30
		elif age==3 or age==3 or age==4:
			return 'age_20_30'

		# Age_30_40
		elif age==5 or age==6 or age==7 or age==8:
			return 'age_30_40'

		# Age_40_50
		elif age==8 or age==9:
			return 'age_40_50'

		# Age_above_50
		else:
			return 'age_above_50'

	# Label Emotion's prediction
	def __decodeEmotion(self, emotion):
		
		if emotion==0:
			return 'Angry'
		elif emotion==1:
			return 'Happy'
		elif emotion==2:
			return 'Sad'
		else:
			return 'Neutral'

	# Build model for emotion classification. 

	###
	#ToDo: Remove this and use .yaml file format for model.
	###
	def __emotion_classification(self, inputs):
	    x = Conv2D(32, (3, 3), padding="same", activation='relu')(inputs)
	    x = Conv2D(32, (3, 3), padding="same", activation='relu')(x)
	    x = MaxPooling2D(pool_size=(2, 2))(x)
	    x = Dropout(0.5)(x)
	                     
	    x = Conv2D(64, (3, 3), padding="same", activation='relu')(x)
	    x = Conv2D(64, (3, 3), padding="valid", activation='relu')(x)
	    x = MaxPooling2D(pool_size=(2, 2))(x)
	    x = Dropout(0.5)(x)
	    
	    x = Conv2D(96, (3, 3), padding="same", activation='relu')(x)
	    x = Conv2D(96, (3, 3), padding="valid", activation='relu')(x)
	    x = MaxPooling2D(pool_size=(2, 2))(x)
	    x = Dropout(0.5)(x)
	    
	    x = Conv2D(128, (3, 3), dilation_rate=(2, 2), padding="same", activation='relu')(x)
	    x = Conv2D(128, (3, 3), padding="valid", activation='relu')(x)
	    x = MaxPooling2D(pool_size=(2, 2))(x)
	    x = Dropout(0.5)(x)
	    
	    x = Flatten()(x)
	    x = Dense(512, activation='relu')(x)
	    x = Dense(128, activation='relu')(x)
	    x = Dense(4 , activation='softmax')(x)
	    
	    model = Model(inputs=inputs, outputs=x)
	    
	    return x

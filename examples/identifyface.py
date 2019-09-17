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

from keras.models import load_model


class IdentifyFace():
	def __init__(self, weight_path=['face_detector/weights/gender-race-age.hdf5', 'face_detector/weights/emotion.hdf5']):

		self._gra_weight_path = weight_path[0]
		self._e_weight_path = weight_path[1]

		self.__gra_model = None
		self.__e_model = None

		self.__graph1 = None
		self.__graph2 = None
		self.__session1 = None
		self.__session2 = None

		print("[INFO] Preparing Model and loading weights...")
		self.gra_buildModel()
		self.e_buildModel()
		print("[INFO] Done")
	
	def gra_buildModel(self):
		self.__graph1 = tf.Graph()
		with self.__graph1.as_default():
			self.__session1 = tf.Session()
			with self.__session1.as_default():
				inputLayer = Input(shape=(200,200,3))
				gender = self.__gender_classificaton(inputLayer)
				ethnicity = self.__ethnicity_classification(inputLayer)
				age = self.__age_classification(inputLayer)

				gra_model = Model(inputs=inputLayer, outputs=[gender, ethnicity, age])
				gra_model = load_model(self._gra_weight_path)
				self.__gra_model = gra_model


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
					gender, ethnicity, age = self.__gra_model.predict(gra_image)

			# print("[CHECKING] Gender: ", np.argmax(gender))
			gender = self.__decodeGender(np.argmax(gender))
			# print("[CHECKING] Ethnicity: ", np.argmax(ethnicity))
			ethnicity = self.__decodeEthnicity(np.argmax(ethnicity))
			# print("[CHECKING] Age: ", np.argmax(age))
			age = self.__degroupAge(np.argmax(age))

			with self.__graph2.as_default():
				with self.__session2.as_default():
					emotion = self.__e_model.predict(e_image)

			# print("[CHECKING] Emotion: ", np.argmax(emotion))
			emotion = self.__decodeEmotion(np.argmax(emotion))

			output[face_id] = [gender, ethnicity, age, emotion]

		print("[INFO] Done")
		return output

	def __decodeGender(self, gender):
		if gender==0:
			return 'male'
		else:
			return 'female'

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

	def __decodeEmotion(self, emotion):
		
		if emotion==0:
			return 'Angry'
		elif emotion==1:
			return 'Happy'
		elif emotion==2:
			return 'Sad'
		else:
			return 'Neutral'

	def __gender_classificaton(self, inputLayer):
	    x = Conv2D(32, kernel_size=(3,3), padding='same', activation='relu')(inputLayer)
	    x = Conv2D(64, kernel_size=(3,3), padding='same', activation='relu')(x)
	    x = MaxPooling2D(pool_size=(2,2))(x)
	    x = BatchNormalization()(x)
	    x = Dropout(0.25)(x)
	    
	    x = Conv2D(128, kernel_size=(3,3), padding='valid', activation='relu')(x)
	    x = Conv2D(128, kernel_size=(3,3), padding='same', activation='relu')(x)
	    x = MaxPooling2D(pool_size=(2,2))(x)
	    x = BatchNormalization()(x)
	    x = Dropout(0.25)(x)
	    
	    x = Conv2D(256, kernel_size=(3,3), padding='valid', activation='relu')(x)
	    x = Conv2D(256, kernel_size=(3,3), padding='same', activation='relu')(x)
	    x = MaxPooling2D(pool_size=(2,2))(x)
	    x = BatchNormalization()(x)
	    x = Dropout(0.25)(x)
	    
	    x = Flatten()(x)
	    x = Dense(256, activation='relu')(x)
	    x = Dense(128, activation='relu')(x)
	    x = Dense(64, activation='relu')(x)
	    x = Dense(1, activation='sigmoid', name='gender')(x)
	    
	    return x

	def __ethnicity_classification(self, inputLayer):
	    x = Conv2D(64, kernel_size=(3,3), padding='same', activation='relu')(inputLayer)
	    x = Conv2D(64, kernel_size=(3,3), padding='same', activation='relu')(inputLayer)
	    x = MaxPooling2D(pool_size=(2,2))(x)
	    x = BatchNormalization()(x)
	    x = Dropout(0.25)(x)
	    
	    x = Conv2D(64, kernel_size=(3,3), padding='valid', activation='relu')(x)
	    x = Conv2D(64, kernel_size=(3,3), padding='same', activation='relu')(x)
	    x = MaxPooling2D(pool_size=(2,2))(x)
	    x = BatchNormalization()(x)
	    x = Dropout(0.25)(x)
	    
	    x = Conv2D(128, kernel_size=(3,3), padding='valid', activation='relu')(x)
	    x = Conv2D(128, kernel_size=(3,3), padding='same', activation='relu')(x)
	    x = MaxPooling2D(pool_size=(2,2))(x)
	    x = BatchNormalization()(x)
	    x = Dropout(0.25)(x)
	    
	    x = Conv2D(256, kernel_size=(3,3), padding='valid', activation='relu')(x)
	    x = Conv2D(256, kernel_size=(3,3), padding='same', activation='relu')(x)
	    x = MaxPooling2D(pool_size=(2,2))(x)
	    x = BatchNormalization()(x)
	    x = Dropout(0.25)(x)
	    
	    x = Flatten()(x)
	    x = Dense(512, activation='relu')(x)
	    x = Dense(256, activation='relu')(x)
	    x = Dense(5, activation='softmax', name='ethnicity')(x)
	    
	    return x


	def __age_classification(self, inputLayer):
	    x = Conv2D(32, kernel_size=(3,3), padding='same', activation='relu')(inputLayer)
	    x = Conv2D(32, kernel_size=(3,3), padding='same', activation='relu')(inputLayer)
	    x = MaxPooling2D(pool_size=(2,2))(x)
	    x = BatchNormalization()(x)
	    x = Dropout(0.25)(x)
	    
	    x = Conv2D(64, kernel_size=(3,3), padding='valid', activation='relu')(x)
	    x = Conv2D(64, kernel_size=(3,3), padding='same', activation='relu')(x)
	    x = MaxPooling2D(pool_size=(2,2))(x)
	    x = BatchNormalization()(x)
	    x = Dropout(0.25)(x)
	    
	    x = Conv2D(128, kernel_size=(3,3), padding='valid', activation='relu')(x)
	    x = Conv2D(128, kernel_size=(3,3), padding='same', activation='relu')(x)
	    x = MaxPooling2D(pool_size=(2,2))(x)
	    x = BatchNormalization()(x)
	    x = Dropout(0.25)(x)
	    
	    x = Conv2D(256, kernel_size=(3,3), padding='valid', activation='relu')(x)
	    x = Conv2D(256, kernel_size=(3,3), padding='same', activation='relu')(x)
	    x = MaxPooling2D(pool_size=(2,2))(x)
	    x = BatchNormalization()(x)
	    x = Dropout(0.25)(x)
	    
	    x = Flatten()(x)
	    x = Dense(1024, activation='relu')(x)
	    x = Dense(512, activation='relu')(x)
	    x = Dense(256, activation='relu')(x)
	    x = Dense(12, activation='softmax', name='age')(x)
	    
	    return x

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

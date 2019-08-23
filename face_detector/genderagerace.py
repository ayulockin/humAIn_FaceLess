import numpy as np 
import pandas as pd
import os
import cv2

import keras.backend as K
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense, Flatten, BatchNormalization, Dropout
from keras.layers import Conv2D, MaxPooling2D

from keras.models import load_model


class GenderRaceAge():
	def __init__(self, weight_path='weights/gender-race-age.hdf5'):

		self._weight_path = weight_path
		self.__model = None
		self.buildModel()
	
	def buildModel(self):
		print("[INFO] Preparing Model...")
		inputLayer = Input(shape=(200,200,3))
		gender = self.__gender_classificaton(inputLayer)
		ethnicity = self.__ethnicity_classification(inputLayer)
		age = self.__age_regression(inputLayer)

		model = Model(inputs=inputLayer, outputs=[gender, ethnicity, age])
		print("[INFO] Done")
		print("[INFO] Loading Weights...")
		model = load_model(self._weight_path)
		self.__model = model
		print("[INFO] Done")

	def predict(self, image_dict):
		print("[INFO] Predicting Gender, Ethnicity, Age...")
		output = {}
		for face_id, image in image_dict.items():
			image = cv2.resize(image, (200,200))
			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			image = image.reshape(((1,)+image.shape))

			gender, ethnicity, age = self.__model.predict(image)
			gender = np.argmax(gender)
			ethnicity = np.argmax(ethnicity)
			age = self.__degroupAge(np.argmax(age))

			output[face_id] = [gender, ethnicity, age]

		print("[INFO] Done")
		return output

	def __degroupAge(self, age):
		# Age_below20
		if age==0 or age==1 or age==2:
			return 'age_below20'

		# Age_20_30
		elif age==3 or age==4 or age==5:
			return 'age_20_30'

		# Age_30_40
		elif age==6 or age==7 or age==8:
			return 'age_30_40'

		# Age_40_50
		elif age==9 or age==10:
			return 'age_40_50'

		# Age_above_50
		else:
			return 'age_above_50'


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


	def __age_regression(self, inputLayer):
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

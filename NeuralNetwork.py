import os
import time
import math
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import seaborn as sns
import numpy as np

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.3333)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

train_path = 'C:/Users/nived/Desktop/programing/Pythonstuff/machine_learning/Practice/PneumoniaXray/train'
test_path = 'C:/Users/nived/Desktop/programing/Pythonstuff/machine_learning/Practice/PneumoniaXray/test'
valid_path = 'C:/Users/nived/Desktop/programing/Pythonstuff/machine_learning/Practice/PneumoniaXray/val'

CATEGORIES = ["normal", "pneumonia"]

img_height = 224
img_width = 224
batch_size = 64

shear_range = 0.2
zoom_range = 0.3
rotation_range = 0
horizontal_flip = False

for category in CATEGORIES:
	path = os.path.join(train_path, category)
	for img in os.listdir(path):
		img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
		new_array = cv2.resize(img_array, (img_width, img_height))
		plt.imshow(new_array, cmap='gray')
		plt.show()
		break
	break


def img_augmentation(shear_range, zoom_range, rotation_range, horizontal_flip):
	global train, test, valid
	image_gen = ImageDataGenerator(rescale=1/255,
			shear_range=shear_range,
			zoom_range=zoom_range,
			horizontal_flip=horizontal_flip)

	test_data_gen = ImageDataGenerator(rescale=1/255)

	train = image_gen.flow_from_directory(
		train_path,
		target_size=(img_height, img_width),
		shuffle=True,
		color_mode='grayscale',
		batch_size=batch_size,
		class_mode='binary',
		)

	test = test_data_gen.flow_from_directory(
		test_path,
		target_size=(img_height, img_width),
		shuffle=False,
		color_mode='grayscale',
		class_mode='binary',
		batch_size=64
		)

	valid = image_gen.flow_from_directory(
		valid_path,
		target_size=(img_height, img_width),
		color_mode='grayscale',
		shuffle=False,
		class_mode='binary',
		batch_size=64
		)


class build_model(object):
	def __init__(self, activation, conv_layer, dense_layer, layer_size):
		self.activation = activation
		self.conv_layer = conv_layer
		self.dense_layer = dense_layer
		self.layer_size = layer_size

	def conv_block(self, filter_size):
		block = [Conv2D(filter_size, (3, 3), activation=self.activation, input_shape=(img_width, img_height, 1), padding='same'),
		MaxPooling2D(pool_size=(2, 2))]

		l = 1
		while l < self.conv_layer:
			block.append(Conv2D(filter_size*(2**l), (3, 3), activation=self.activation, padding='same')),
			block.append(MaxPooling2D(pool_size=(1+l, 1+l)))

			l = l + 1

		return block

	def dense_block(self):
		block=[Dense(activation=self.activation, units=self.layer_size)]

		l = 2
		while l < self.dense_layer+1:
			block.append(Dense(activation=self.activation, units=self.layer_size/l))
			l = l + 2

			return block

def model_add(filter_size, activation, dropout_rate):
	global model

	CONV_LAYER = 3
	LAYER_SIZE = 128
	DENSE_LAYER = 2

	NN = build_model(activation, CONV_LAYER, DENSE_LAYER, LAYER_SIZE)

	model = Sequential([
		NN.conv_block(filter_size)[0],
		NN.conv_block(filter_size)[1],
		NN.conv_block(filter_size)[2],
		NN.conv_block(filter_size)[3],
		NN.conv_block(filter_size)[4],
		NN.conv_block(filter_size)[5],
		Flatten(),
		NN.dense_block()[0],
		NN.dense_block()[1],
		Dropout(rate=0.2),
		Dense(activation='sigmoid', units=1)])

	return model


class model_eval(object):
	def __init__(self, train, test, val, epochs):
		self.train = train
		self.test = test
		self.val = val
		self.epochs = epochs

	def balance_classes(self):
		weights = compute_class_weight('balanced', np.unique(train.classes), train.classes)
		cw = dict(zip(np.unique(train.classes), weights))

		return cw


	def callback_list(self):
		early = EarlyStopping(monitor='val_accuracy', patience=5)

		learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss',
			patience=1, verbose=1, factor=0.5, min_lr=0.000001)

		checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("model.h5", 
			save_best_only=True)

		callback_list = [early, learning_rate_reduction]

		return callback_list


	def train_test(self, optimizer):
		global history
		model_add(32, 'relu', 0.2)
		model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
		model.summary()

		cw = self.balance_classes()
		callbacks_list = self.callback_list()

		history = model.fit(self.train, epochs=self.epochs, validation_data=self.val, class_weight=cw, callbacks=callbacks_list)

		test_accuracy = model.evaluate(self.test)
		print('The testing accuracy is: ', test_accuracy[1]*100, '%')

		return history


def main():
	img_augmentation(shear_range, zoom_range, rotation_range, horizontal_flip)
	epochs = 50

	eval = model_eval(train, test, valid, epochs)
	eval.train_test('adam')

	accuracy = history.history['accuracy']
	val_accuracy = history.history['val_accuracy']
	plt.plot(range(len(accuracy)), accuracy, color='blue', label='Training accuracy')
	plt.plot(range(len(accuracy)), val_accuracy, color='red', label='Validation accuracy')
	plt.xlabel('Epoch No.')
	plt.ylabel('Accuracy')
	plt.legend()
	plt.show()

	loss = history.history['loss']
	val_loss = history.history['val_loss']
	plt.plot(range(len(accuracy)), loss, color='blue', label='Training loss')
	plt.plot(range(len(accuracy)), val_loss, color='red', label='Validation loss')
	plt.xlabel('Epoch No.')
	plt.ylabel('loss')
	plt.legend()
	plt.show()

	true = test.classes
	preds = model.predict(test, verbose=1)
	predictions = preds.copy()
	predictions[predictions <= 0.5] = 0
	predictions[predictions > 0.5] = 1

	cm = confusion_matrix(true, np.round(predictions))
	fig, ax = plt.subplots()
	fig.set_size_inches(12, 8)
	sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap=plt.cm.Blues, cbar=False)
	ax.tick_params(axis='x', labelsize=16)
	ax.tick_params(axis='y', labelsize=16)
	ax.set_ylabel("True", color="royalblue", fontsize=35, fontweight=700)
	ax.set_xlabel("Prediction", color="royalblue", fontsize=35, fontweight=700)
	plt.yticks(rotation=0)
	plt.show()

	print(classification_report(y_true=test.classes, y_pred=predictions, target_names=['NORMAL', 'PNEUMONIA']))

if __name__ == '__main__':
	main()

'''
model = model.load_model('model.h5')

def prepare(filepath):
	IMG_SIZE = 224
	img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
	new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
	return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

prediction = model.predict([prepare('pneumonia.jpg')])
print(CATEGORIES[int(prediction[0][0])])
'''
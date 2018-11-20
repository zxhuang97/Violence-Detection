import os
import sys
import random

import keras
import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from model import build_model
from transform import trainAug,valAug
from dataloader import DataGenerator
from keras.losses import binary_crossentropy
import keras.backend.tensorflow_backend as KTF
from keras import optimizers
KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu':0})))

class ViolenceNet:
	def __init__(self):

		self.epoch =150
		self.lr =0.005
		self.lr_drop = 5
		self.batch_size = 1
		self.model = build_model()
		self.trainLoader = DataGenerator('./data/rgb_train.txt', batch_size=self.batch_size, random_shift = True, transform = trainAug())
		self.valLoader = DataGenerator('./data/rgb_val.txt', batch_size=self.batch_size, random_shift = False, transform = valAug())

	def train(self):
		opti = optimizers.SGD(lr=self.lr,momentum=0.9, nesterov=True)
		self.model.compile(loss = 'binary_crossentropy',optimizer = opti,metrics=['accuracy'])


		def classifier_lr_scheduler(epoch):
			return self.lr * (0.5 ** (epoch // self.lr_drop))
		classifier_reduce_lr = keras.callbacks.LearningRateScheduler(classifier_lr_scheduler)



		self.model.fit_generator(self.trainLoader,max_queue_size = 1 ,workers=1,steps_per_epoch=640//self.batch_size,epochs=self.epoch,verbose=1,
			validation_data=self.valLoader, callbacks=[classifier_reduce_lr])



if __name__ == '__main__':

	net = ViolenceNet()
	net.train()


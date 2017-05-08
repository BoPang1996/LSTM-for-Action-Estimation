from keras.models import Sequential
from keras.layers import LSTM, Dense
import tensorflow as tf

def buildmodel(opt):
	print('Build model...')
	with tf.device('/gpu:1'):
		model = Sequential()
		model.add(LSTM(512, input_shape=(None,1000)))#dropout_W=0.2, dropout_U=0.2
		model.add(Dense(opt.nClasses, activation='softmax'))

		# try using different optimizers and different optimizer configs
		print('Compile model...')
		model.compile(loss='categorical_crossentropy',
					optimizer='adam',
					metrics=['accuracy'])
		print('Compile finished.')
				
		return model


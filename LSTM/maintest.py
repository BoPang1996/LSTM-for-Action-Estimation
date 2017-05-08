from LSTMmodel import buildmodel
from train import train
from test import test, load_model
import numpy as np
import keras
import tensorflow as tf

class Data:
	def __init__(self):
		self.features=[]
		self.label=[]

class Opt:
	def __init__(self):
		self.nClasses = 157
		self.batchSize = 64
		self.nEpochs = 5
		self.train = 3  # 1:train   2:test     3:resumeTrain
		self.modelpath = "TrainedModel/"
		self.datapath = "data/"

opts = Opt()

if (opts.train==1):
	#data_train = readdata(opts.datapath + "trainfeatures", opts.datapath + "trainlabels.csv")
	data_train = Data()
	data_train.features = np.random.random((100, 100, 1000))
	data_train.label = keras.utils.to_categorical(np.random.randint(157, size=(100, 1)), num_classes=157)
	model = buildmodel(opts)
	train(model, opts, data_train)
if (opts.train==2):
	#data_test = readdata(opts.datapath + "testfeatures", opts.datapath + "testlabels.csv")
	data_test = Data()
	data_test.features = np.random.random((100, 100, 1000))
	data_test.label = keras.utils.to_categorical(np.random.randint(157, size=(100, 1)), num_classes=157)
	test(opts, data_test)
if(opts.train==3):
	data_train = Data()
        data_train.features = np.random.random((100, 100, 1000))
        data_train.label = keras.utils.to_categorical(np.random.randint(157, size=(100, 1)), num_classes=157)
	with tf.device('/gpu:1'):
		model = load_model(opts.modelpath+"struct_file", opts.modelpath+"weights_file")
		train(model, opts, data_train)

from LSTMmodel import buildmodel
from train import train
from test import test
from readdata import Data,readdata


class Opt:
	def __init__(self):
		self.nClasses=157
		self.batchSize=64
		self.nEpochs=25
		self.train=True  #true:train   false:test
		self.modelpath="TrainedModel/"
		self.datapath="data/"
		
opts = Opt()

if opts.train:
	data_train = readdata(opts.datapath+"trainfeatures",opts.datapath+"trainlabels.csv")
	model = buildmodel(opts)
	train(model, opts, data_train)
else:
	data_test = readdata(opts.datapath+"testfeatures",opts.datapath+"testlabels.csv")
	test(opts,data_test)
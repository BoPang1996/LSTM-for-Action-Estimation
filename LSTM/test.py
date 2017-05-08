from keras.models import Sequential
from keras.models import model_from_json
import tensorflow as tf
def test(opt, data):
	with tf.device('/gpu:1'):
		model=load_model(opt.modelpath+"struct_file", opt.modelpath+"weights_file")
		score, acc = model.evaluate(data.features, data.label, batch_size=opt.batchSize)
		print('Test score:', score)
		print('Test accuracy:', acc)
		output = Sequential.predict(model,data.features, batch_size=opt.batchSize, verbose=0)
		
		return output
	
	
def load_model(struct_file, weights_file):
	model = model_from_json(open(struct_file, 'r').read())
	model.compile(loss="categorical_crossentropy", optimizer='adam',metrics=['accuracy'])
	model.load_weights(weights_file)
	return model

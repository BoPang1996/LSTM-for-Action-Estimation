def train(model,opt,data):
	print('Train...')
	model.fit(data.features, data.label, batch_size=opt.batchSize, epochs=opt.nEpochs) #validation_data=(data.features, data.label)
	#score, acc = model.evaluate(data.features, data.label, batch_size=opt.batchSize)
	#print('Test score:', score)
	#print('Test accuracy:', acc)
	print("Training Finish.")
	
	save_model_to_file(model, opt.modelpath+"struct_file", opt.modelpath+"weights_file")
	print("Model Saved.")

def save_model_to_file(model, struct_file, weights_file):
    # save model structure
    model_struct = model.to_json()
    open(struct_file, 'w').write(model_struct)

    # save model weights
    model.save_weights(weights_file, overwrite=True)
	
	


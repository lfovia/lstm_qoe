import os
from network_init import network_init

##################################################################
def network_train(no_LSTM_layers,no_LSTM_units,X,Y,no_epochs,batch_size,timesteps,no_feat,loss,opt,X_valid,Y_valid):    
    
	# Initialization
	min_epochs = 5
	best_epoch_no = 1    
	i = 1

	# Initialize the LSTM network    
	net_ts = network_init(no_LSTM_layers,no_LSTM_units,batch_size,timesteps,no_feat,loss,opt)    
    
    # Train over the specified no. of epochs
	while i <= no_epochs:

		# Train the network
		history = net_ts.fit(X,Y,nb_epoch=1,batch_size=batch_size,verbose=2,shuffle=False,validation_data=(X_valid, Y_valid))
		current_valid_loss = history.history['val_loss']
		current_training_loss = history.history['loss']   

		print 'current epoch_no:', i            

		# track the validation loss	
		if i == min_epochs:
			best_valid_loss = current_valid_loss
			best_epoch_no = i

			# save model weights
			net_ts.save_weights(filepath='model_weights_temp/best_model_weights.hdf5',overwrite=True)

		if i > min_epochs:			
			if current_valid_loss < best_valid_loss:
				best_valid_loss = current_valid_loss
				best_epoch_no = i					

				# save model weights
				net_ts.save_weights(filepath='model_weights_temp/best_model_weights.hdf5',overwrite=True)		

			print 'best_epoch_no', best_epoch_no
			print 'best_valid_loss', best_valid_loss			    

		# reset the LSTM states
		net_ts.reset_states()	    	    
		i = i + 1

	net = network_init(no_LSTM_layers,no_LSTM_units,batch_size,1,no_feat,loss,opt)    
	net.load_weights(filepath='model_weights_temp/best_model_weights.hdf5')
	os.remove('model_weights_temp/best_model_weights.hdf5')
	
	return net
from keras.models import Sequential
from keras.layers.core import Dense,Activation
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed

################################################
def network_init(no_LSTM_layers,no_LSTM_units,batch_size,timesteps,no_feat,loss,opt):

	# define the network
    net = Sequential()
    net.add(LSTM(no_LSTM_units,batch_input_shape=(batch_size, timesteps, no_feat),return_sequences = True, stateful = True))
    
    if no_LSTM_layers > 1:
        for ii in range(no_LSTM_layers-1):
            net.add(LSTM(no_LSTM_units, return_sequences = True, stateful = True))
   
    net.add(TimeDistributed(Dense(1,activation='relu')))
    net.compile(loss=loss,optimizer=opt)
    
    return net
# LSTM-QoE code for LIVE Mobile Stall II Database

import os
import shutil
import keras 
import numpy as np
from numpy import random as rnd

from sklearn.metrics import mean_squared_error
from math import sqrt

import scipy.io as sio
from scipy.io import loadmat, savemat
from scipy.stats.stats import pearsonr 
from scipy.stats import spearmanr

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import time
import warnings

from network_train import network_train
from OR import OR

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Main Program %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

warnings.filterwarnings('ignore')

# start the timer
tic = time.clock()

# Initialize the random seed
rnd.seed(4)

############# Inputs ############

path1 = "LIVE_Mobile_Stall_II_matfiles/"

path2 = "pred_QoE_matfiles/"
path3 = "plots/"
path4 = "model_weights_temp/"

if not os.path.exists(path2):
    os.mkdir(path2)

if not os.path.exists(path3):
    os.mkdir(path3)

if not os.path.exists(path4):
    os.mkdir(path4)

dataset_name = 'LIVE_Mobile_Stall_II'

# loss function
loss = 'mse'

# optimizer
opt = 'adam'

# LSTM network initializations
no_LSTM_layers = 2
no_LSTM_units = 22

timesteps = 4
no_epochs = 50
batch_size = 1 # batch size fixed to 1 as the LSTM network is stateful

# Dataset information
no_videos = 174
no_training_videos = 140
no_validation_videos = 33

# no. of input features
no_feat = 2

# Normalizing factors
score_continuous_max = 100
score_continuous_min = 0

# Load training and validation sets
Stall_DB_more_EXOG_content = sio.loadmat(path1+'Stall_DB_more_EXOG.mat')
Nrebuffers_all = np.transpose(Stall_DB_more_EXOG_content['Nrebuffers'])
TSL_all = np.transpose(Stall_DB_more_EXOG_content['TSL'])

vidStallData_content = sio.loadmat(path1+'vidStallData.mat')
continuousQoE_all = vidStallData_content['continuousQoE_s']
continuousQoE_CIhigh_all = vidStallData_content['continuousQoE_s_CIhigh']
continuousQoE_CIlow_all = vidStallData_content['continuousQoE_s_CIlow']

# Initializations
LCC_test = []    
SROCC_test =[]
RMSE_test = []
OR_test = []


################# Begin evaluation ##############################

# create test video indexes
all_videos = range(1,no_videos+1)
all_videos = np.array(all_videos)

for test_video_no in range(1,no_videos+1):

	aa = all_videos[0:test_video_no-1]
	bb = all_videos[test_video_no:,]

	# collect all video indexes except the test video index
	TrVaV = np.hstack((aa,bb))

	# permutate
	perm_videos = np.random.permutation(TrVaV)

	# define the training and validation sets
	TrV = perm_videos[0:no_training_videos]
	VaV = perm_videos[no_training_videos:,]

	training_features = []
	training_features = np.array(training_features)

	training_labels = []
	training_labels = np.array(training_labels)

	print('test video: %d'%(test_video_no))
  	  	

	################## Training #######################

	for ii in range(no_training_videos):

		video_features = []
		video_features = np.array(video_features)

		training_video_no = TrV[ii]

		Nrebuffers = np.transpose(Nrebuffers_all[training_video_no-1,0])
		TSL = np.transpose(TSL_all[training_video_no-1,0])				

		score_continuous = np.transpose(continuousQoE_all[training_video_no-1,0])
		score_continuous_norm = (score_continuous - score_continuous_min)/(score_continuous_max - score_continuous_min)

		video_features = Nrebuffers
		video_features = np.hstack((video_features,TSL))

		# constitute the input feature vector and the output QoE vector
		if ii == 0:
			training_features = video_features	
			training_labels = score_continuous_norm		
		else:
			training_features = np.vstack((training_features,video_features))
			training_labels = np.vstack((training_labels,score_continuous_norm))
	
	
	# reshape for timesteps
	training_features2 = training_features[0:len(training_features)/timesteps *timesteps,:]
	training_labels2 = training_labels[0:len(training_labels)/timesteps *timesteps,:]

	training_features_ts = np.reshape(training_features2,(-1,timesteps,training_features2.shape[1]))
	training_labels_ts = np.reshape(training_labels2,(-1,timesteps,training_labels2.shape[1]))


	############# Validation ###################	

	for ii in range(no_validation_videos):

		video_features = []
		video_features = np.array(video_features)

		validation_video_no = VaV[ii]

		Nrebuffers = np.transpose(Nrebuffers_all[validation_video_no-1,0])
		TSL = np.transpose(TSL_all[validation_video_no-1,0])
		
		score_continuous = np.transpose(continuousQoE_all[validation_video_no-1,0])
		score_continuous_norm = (score_continuous - score_continuous_min)/(score_continuous_max - score_continuous_min)
		
		video_features = Nrebuffers
		video_features = np.hstack((video_features,TSL))		

		if ii == 0:
			validation_features = video_features	
			validation_labels = score_continuous_norm		
		else:
			validation_features = np.vstack((validation_features,video_features))
			validation_labels = np.vstack((validation_labels,score_continuous_norm))		


		validation_features2 = validation_features[0:len(validation_features)/timesteps *timesteps,:]
		validation_labels2 = validation_labels[0:len(validation_labels)/timesteps *timesteps,:]

		validation_features_ts = np.reshape(validation_features2,(-1,timesteps,validation_features2.shape[1]))
		validation_labels_ts = np.reshape(validation_labels2,(-1,timesteps,validation_labels2.shape[1]))
	

	############# Testing ######################
	
	Nrebuffers = np.transpose(Nrebuffers_all[test_video_no-1,0])
	TSL = np.transpose(TSL_all[test_video_no-1,0])	

	score_continuous = np.transpose(continuousQoE_all[test_video_no-1,0])
	score_continuous_CIhigh = np.transpose(continuousQoE_CIhigh_all[test_video_no-1,0])
	score_continuous_CIlow = np.transpose(continuousQoE_CIlow_all[test_video_no-1,0])

	score_continuous_norm = (score_continuous - score_continuous_min)/(score_continuous_max - score_continuous_min)
	score_continuous_CIhigh_norm = (score_continuous_CIhigh - score_continuous_min)/(score_continuous_max - score_continuous_min)
	score_continuous_CIlow_norm = (score_continuous_CIlow - score_continuous_min)/(score_continuous_max - score_continuous_min)       

	epsilon_test = np.asarray(score_continuous_CIhigh_norm - score_continuous_CIlow_norm)
	
	video_features = Nrebuffers	
	video_features = np.hstack((video_features,TSL))

	test_features = video_features
	actual_QoE = score_continuous_norm
		
	test_features = np.reshape(test_features,(-1,1,test_features.shape[1]))


	################## Fit model #######################

	net = network_train(no_LSTM_layers,no_LSTM_units,training_features_ts,training_labels_ts,no_epochs,batch_size,timesteps,no_feat,loss,opt,validation_features_ts,validation_labels_ts)
	
	# prediction using the trained model
	pred_QoE = net.predict(test_features, batch_size=1)
	pred_QoE = np.reshape(pred_QoE,(-1,1))

	test_video_filename = path2+'video_'+str(test_video_no)

	# save the predicted QoE
	sio.savemat(test_video_filename,{
	'pred_QoE': pred_QoE,    
	})
	
	
	# plot the predicted QoE and the actual QoE
	ax = plt.gca()
	plt.plot(pred_QoE,'r')
	plt.plot(actual_QoE,'b')
	ax.legend(('Predicted QoE', 'Actual QoE'))
	plt.savefig(path3+'video_'+str(test_video_no)+'.png')
	plt.close()	
  
	if test_video_no == 1:
	    LCC_temp = pearsonr(pred_QoE,actual_QoE)
	    LCC_test = LCC_temp[0]
	    SROCC_temp = spearmanr(pred_QoE,actual_QoE)
	    SROCC_test = SROCC_temp[0]
	    RMSE_test = sqrt(mean_squared_error(pred_QoE,actual_QoE))
	    OR_test =  OR(pred_QoE,actual_QoE,epsilon_test) 
	    OR_test = np.array(OR_test)
	else:
	    LCC_temp = pearsonr(pred_QoE,actual_QoE)
	    LCC_test = np.append(LCC_test,LCC_temp[0])
	    SROCC_temp = spearmanr(pred_QoE,actual_QoE)
	    SROCC_test = np.append(SROCC_test,SROCC_temp[0])
	    RMSE_test = np.append(RMSE_test,sqrt(mean_squared_error(pred_QoE,actual_QoE)))
	    OR_temp = OR(pred_QoE,actual_QoE,epsilon_test)
	    OR_test = np.append(OR_test,OR_temp)

	# print the cumulative performance so far
	print('Test video: %d' %(test_video_no))
	print('LCC_test_mean:')
	print(np.nanmean(LCC_test))
	print('SROCC_test_mean:')
	print(np.nanmean(SROCC_test))
	print('RMSE_test_mean:')
	print(np.nanmean(RMSE_test))
	print('OR_test_mean:')
	print(np.nanmean(OR_test))

    
print('####################### Test Performance #######################')

print('LCC_test_mean: %0.4f' %np.nanmean(LCC_test))
print('LCC_test_median: %0.4f' %np.median(LCC_test))
print('SROCC_test_mean: %0.4f' %np.nanmean(SROCC_test))
print('SROCC_test_median: %0.4f' %np.median(SROCC_test))
print('RMSE_test_mean: %0.4f' %np.nanmean(RMSE_test))
print('RMSE_test_median: %0.4f' %np.median(RMSE_test))
print('OR_test_mean: %0.4f' %np.nanmean(OR_test))
print('OR_test_median: %0.4f' %np.median(OR_test))

saving_filename = 'LSTM_QoE_'+dataset_name+'_LSTMlayers'+str(no_LSTM_layers)+'_LSTMunits'+str(no_LSTM_units)+'_'+'epochs'+str(no_epochs)+'_'+loss+'_'+opt

sio.savemat(saving_filename,{

'LCC_test': LCC_test,
'SROCC_test': SROCC_test,
'RMSE_test': RMSE_test,
'OR_test': OR_test,

})

if os.path.exists(path4):
	shutil.rmtree(path4)

toc = time.clock()

print('time elapsed is %0.2f seconds' %(toc-tic))

import numpy as np

def find_TrV_index(video_no):	

	no_playout_patterns = 8.0

	a = int(np.ceil(video_no/no_playout_patterns))
	b = int(np.mod(video_no,no_playout_patterns)) - 1;

	if b == -1:
	    b = 7

	return a, b
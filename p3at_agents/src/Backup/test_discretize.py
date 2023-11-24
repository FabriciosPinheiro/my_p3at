import numpy as np

data = []

for i in range(270):
	data.append(i)

def discretize_observation(data,new_ranges):
	discretized_ranges = []
	min_range = 0.02
	done = False
	mod = len(data)/new_ranges
	for i, item in enumerate(data):
		if (i%mod==0):
			if data == float ('Inf'):
				discretized_ranges.append(6)
			elif np.isnan(data[i]):
				discretized_ranges.append(0)
			else:
				discretized_ranges.append(round(data[i],2))
		if (min_range > data[i] > 0):
			done = True

	return discretized_ranges

def discretize_obs(data,new_ranges):
	discretized_ranges = []

	index=np.linspace(0, len(data), new_ranges, dtype = np.int32)
	x=0
	for i in data:
		if i==index[x]:
			discretized_ranges.append(round(data[i],2))
			x+=1
		else:
			if (i+1)==len(data):
				discretized_ranges.append(round(data[i],2))
				
		
	return discretized_ranges

array_laser = discretize_obs(data, 5)

print(array_laser)
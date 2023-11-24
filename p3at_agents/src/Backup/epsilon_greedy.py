import numpy as np

epsilon_min = 0.01
epsilon = 1
epsilon_decay = 0.001
decay_step = 100

for i in range(100):
	epsilon = epsilon_min + (epsilon - epsilon_min) * np.exp(-epsilon_decay * i)

	print(epsilon)
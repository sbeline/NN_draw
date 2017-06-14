import numpy as np
import scipy.io as sio

class NeuronNetwork(object):
	content = sio.loadmat('ex3data1.mat')
	weights = sio.loadmat('ex3weights.mat')
	print(content['X'].shape)
	print(weights['Theta1'].shape)
	print(weights['Theta2'].shape)
	print(weights['Theta1'][0])
	X = content['X']
	y = content['y']
	def __init__(self):
		self.InputLayer = len(X[0])
		self.OutputLayer = 10
		self.HidenLayer = 25
	def propagate(self):
		pass

def sigmoid(z):
	return 1 / (1 + np.exp(-z))

NN = NeuronNetwork()
print(NN.InputLayer)

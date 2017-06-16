import numpy as np
import scipy.io as sio

class NeuronNetwork(object):
	content = sio.loadmat('ex3data1.mat')
	weights = sio.loadmat('ex3weights.mat')
	X = np.matrix(content['X'])
	y = content['y']
	X = np.concatenate((np.ones((X[:,1].size, 1)), X) , axis=1)
	def __init__(self):
		self.InputLayer = len(X[0])
		self.OutputLayer = 10
		self.HidenLayer = 26
		self.Theta1 = np.matrix(weights['Theta1'])
		self.Theta2 = np.matrix(weights['Theta2'])
	def Propagate(self, X):
		z = sigmoid(self.Theta1 * X.T)
		return (z)
	def sigmoid(z):
		return 1 / (1 + np.exp(-z))

NN = NeuronNetwork()
print(NN.X.shape)
a2 = NN.Propagate(NN.X)
print(a2.shape)

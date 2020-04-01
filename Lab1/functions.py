import numpy as np


def softmax(s):
	""" Standard definition of the softmax function """
	return np.exp(s) / np.sum(np.exp(s), axis=0)


def LoadBatch(filename):
	""" Copied from the dataset website """
	import pickle
	with open('Dataset/' + filename, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
		dict[b'data'] = dict[b'data'].astype(float)

	return dict



def computeAccuracy(X, y, W, b):

	P = evaluateClassifier(X, W, b)

	y_pred = np.argmax(P, 0)

	number_of_correct = np.sum(y_pred == y)

	N = X.shape[1]

	acc = number_of_correct / N

	return acc

def lossFunction(Y, P):

	l = - Y * np.log(P)

	return l

def ComputeCost(X, Y, W, b_try, lamda):
	P = evaluateClassifier(X, W, b_try)
	l = lossFunction(Y, P)
	N = X.shape[1]
	W_transposed = np.transpose(W)

	J = 1/N * np.sum(l) + lamda * np.sum(np.dot(W, W_transposed))

	return J


def ComputeGradsNum(X, Y, P, W, b, lamda, h):
	""" Converted from matlab code """
	no = W.shape[0]
	d = X.shape[0]

	grad_W = np.zeros(W.shape);
	grad_b = np.zeros((no, 1));

	c = ComputeCost(X, Y, W, b, lamda);

	for i in range(len(b)):
		b_try = np.array(b)
		b_try[i] += h
		c2 = ComputeCost(X, Y, W, b_try, lamda)
		grad_b[i] = (c2 - c) / h

	for i in range(W.shape[0]):
		for j in range(W.shape[1]):
			W_try = np.array(W)
			W_try[i, j] += h
			c2 = ComputeCost(X, Y, W_try, b, lamda)
			grad_W[i, j] = (c2 - c) / h

	return [grad_W, grad_b]

def ComputeGradsNumSlow(X, Y, P, W, b_try, lamda, h):
	""" Converted from matlab code """
	no = W.shape[0]
	d = X.shape[0]

	grad_W = np.zeros(W.shape)
	grad_b = np.zeros((no, 1))

	for i in range(len(b)):
		b_try = np.array(b)
		b_try[i] -= h
		c1 = ComputeCost(X, Y, W, b_try, lamda)

		b_try = np.array(b)
		b_try[i] += h
		c2 = ComputeCost(X, Y, W, b_try, lamda)

		grad_b[i] = (c2 - c1) / (2 * h)

	for i in range(W.shape[0]):
		for j in range(W.shape[1]):
			W_try = np.array(W)
			W_try[i, j] -= h
			c1 = ComputeCost(X, Y, W_try, b, lamda)

			W_try = np.array(W)
			W_try[i, j] += h
			c2 = ComputeCost(X, Y, W_try, b, lamda)

			grad_W[i, j] = (c2 - c1) / (2 * h)

	return [grad_W, grad_b]


def montage(W):
	""" Display the image for each label in W """
	import matplotlib.pyplot as plt
	fig, ax = plt.subplots(2, 5)
	for i in range(2):
		for j in range(5):
			im = W[i + j, :].reshape(32, 32, 3, order='F')
			sim = (im - np.min(im[:])) / (np.max(im[:]) - np.min(im[:]))
			sim = sim.transpose(1, 0, 2)
			ax[i][j].imshow(sim, interpolation='nearest')
			ax[i][j].set_title("y=" + str(5 * i + j))
			ax[i][j].axis('off')
	plt.show()


def normalize(train_data, valid_data, test_data):
	mean = np.mean(train_data, 1)
	std = np.std(train_data, 1)

	train_data = np.transpose(train_data) - mean
	train_data = train_data / std

	valid_data = np.transpose(valid_data) - mean
	valid_data = valid_data / std

	test_data = np.transpose(test_data) - mean
	test_data = test_data / std

	return train_data, valid_data, test_data


def save_as_mat(data, name="model"):
	""" Used to transfer a python model to matlab """
	import scipy.io as sio
	sio.savemat(name + '.mat', {name: b})


def evaluateClassifier(X, W, b):
	z = np.dot(W, X)
	s = z + b
	P = softmax(s)

	return P

def generate_mini_batches(X, Y, batch_size):
	# TODO: solve non-even division
	n_batches = int(X.shape[1] / batch_size)

	x_batches = np.zeros((n_batches, X.shape[0], batch_size))
	y_batches = np.zeros((n_batches, Y.shape[0], batch_size))

	for i in range(n_batches):
		x_batches[i] = X[:,i*batch_size: (i+1) * batch_size]
		y_batches[i] = Y[:,i * batch_size: (i+1) * batch_size]

	return x_batches, y_batches

def ComputeGrads(X, Y, P, W, lamda, batch_size):
	G = - (Y - P)
	dl_dw = (1 / batch_size) * np.dot(G, np.transpose(X))
	grad_b = (1 / batch_size) * np.sum(G,1)

	grad_W = dl_dw + 2*lamda*W

	return [grad_W, grad_b]

def one_hot_matrix(Y, K):
	Y_hot = np.zeros((len(Y), K))
	Y_hot[np.arange(len(Y)),Y] = 1
	Y_hot = np.transpose(Y_hot)

	return Y_hot

def minibatchGD(X, Y, GDparams, W, b, lamda):

	x_batches, y_batches = generate_mini_batches(X, Y, GDparams["batch_size"])

	for i in range(GDparams["n_epochs"]):
		for j in range (len(y_batches)):
			P_batch = evaluateClassifier(x_batches[j], W, b)

			grad_W, grad_b = ComputeGrads(x_batches[j], y_batches[j], P_batch, W, lamda, GDparams["batch_size"])

			W -= GDparams["eta"] * grad_W
			b -= np.matrix(GDparams["eta"] * grad_b).T
		J = ComputeCost(x_batches[j], y_batches[j], W, b, lamda)
		print(J)
	return W, b

def main():
	# Load data
	training_data = LoadBatch("data_batch_1")
	validation_data = LoadBatch("data_batch_2")
	test_data = LoadBatch("test_batch")

	# Normalize data
	training_data[b'data'], validation_data[b'data'], test_data[b'data'] = \
		normalize(training_data[b'data'], validation_data[b'data'], test_data[b'data'])

	# Set seed of random generator
	# TODO: Remove when done
	np.random.seed(seed=4540)

	# Initialize parameters

	d = np.shape(training_data[b'data'])[0]
	K = len(np.unique(training_data[b'labels']))
	W = np.random.normal(0.0, 0.01, (K, d))
	b = np.random.normal(0.0, 0.01, (K, 1))
	lamda = 0
	GDparams = {"batch_size": 100, "eta": 0.001, "n_epochs": 20}

	X = training_data[b'data']
	Y_batch_one_hot = one_hot_matrix(training_data[b'labels'], K)
	W, b = minibatchGD(X, Y_batch_one_hot, GDparams, W, b, lamda)

	acc = computeAccuracy(training_data[b'data'], training_data[b'labels'], W, b)

	print(acc)


main()

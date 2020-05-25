import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def softmax(x):
    """ Standard definition of the softmax function """
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def LoadBatch(filename):
    """ Copied from the dataset website """
    import pickle
    with open('Dataset/'+ filename, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
        dict[b'data'] = dict[b'data'].astype(float)
    return dict

def ComputeGradsNum(X, Y, W, b, lamda, h):
    """ Converted from matlab code """
    no 	= 	W.shape[0]
    d 	= 	X.shape[0]

    grad_W = np.zeros(W.shape)
    grad_b = np.zeros((no, 1))

    c = ComputeCost(X, Y, W, b, lamda);

    for i in range(len(b)):
        b_try = np.array(b)
        b_try[i] += h
        c2 = ComputeCost(X, Y, W, b_try, lamda)
        grad_b[i] = (c2-c) / h

    for i in range(W.shape[0]):
    	for j in range(W.shape[1]):
    		W_try = np.array(W)
    		W_try[i,j] += h
    		c2 = ComputeCost(X, Y, W_try, b, lamda)
    		grad_W[i,j] = (c2-c) / h

    return [grad_W, grad_b]

def ComputeGradsNumSlow(X, Y, W, b, lamda, h):
    """ Converted from matlab code """
    no 	= 	W.shape[0]
    d 	= 	X.shape[0]

    grad_W = np.zeros(W.shape)
    grad_b = np.zeros((no, 1))

    for i in range(len(b)):
    	b_try = np.array(b)
    	b_try[i] -= h
    	c1 = ComputeCost(X, Y, W, b_try, lamda)

    	b_try = np.array(b)
    	b_try[i] += h
    	c2 = ComputeCost(X, Y, W, b_try, lamda)

    	grad_b[i] = (c2-c1) / (2*h)

    for i in range(W.shape[0]):
    	for j in range(W.shape[1]):
    		W_try = np.array(W)
    		W_try[i,j] -= h
    		c1 = ComputeCost(X, Y, W_try, b, lamda)

    		W_try = np.array(W)
    		W_try[i,j] += h
    		c2 = ComputeCost(X, Y, W_try, b, lamda)

    		grad_W[i,j] = (c2-c1) / (2*h)

    return [grad_W, grad_b]

def montage(W):
    """ Display the image for each label in W """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2,5)
    for i in range(2):
    	for j in range(5):
    		im  = W[5*i+j,:].reshape(32,32,3, order='F')
    		sim = (im-np.min(im[:]))/(np.max(im[:])-np.min(im[:]))
    		sim = sim.transpose(1,0,2)
    		ax[i][j].imshow(sim, interpolation='nearest')
    		ax[i][j].set_title("y="+str(5*i+j))
    		ax[i][j].axis('off')
    plt.show()

def save_as_mat(data, name="model"):
    """ Used to transfer a python model to matlab """
    import scipy.io as sio
    sio.savemat(name + '.mat', {name:b})

# ----------- My Own down below ----------------

def EvaluateClassifier(X, W, b):
    s = np.dot(W,X) + b # Forward pass
    P = softmax(s)
    return P

def ComputeCost(X, Y, W, b, lamb):
    P = EvaluateClassifier(X, W, b)
    n = X.shape[1]
    loss = - Y * np.log(P)
    loss = (np.sum(loss) / n)
    regularization = lamb * np.sum(W**2)
    J = loss + regularization

    return J, loss

def ComputeAccuracy(X, Y, W, b): # Should be 10 % if not trained since there are 10 classes
    P = EvaluateClassifier(X, W, b)
    P = np.argmax(P,0)
    Y = np.argmax(Y,0)
    acc = np.sum(P == Y) / len(Y)

    return acc

def ComputeGradients(X, Y, W, b, lamb):
    K = W.shape[0]
    d = X.shape[0]
    n = X.shape[1]

    P = EvaluateClassifier(X, W, b)
    G = - (Y - P)

    grad_W = ((1/n) * np.dot(G, X.transpose())) + 2 * lamb * W
    grad_b = (1/n) * np.dot(G, np.ones((n, 1)))

    return grad_W, grad_b

def MiniBatchGD(trainX, trainY, validationX, validationY, testX, testY, GDparams, W, b, lamb):
    costs_train = []
    costs_valid = []
    accs_train = []
    accs_valid = []
    losses_train = []
    losses_valid = []

    n = trainX.shape[1]
    for i in range(GDparams['n_epochs']):

        for j in range(1, int(n / GDparams['n_batch'])):

            j_start = (j-1) * GDparams['n_batch']
            j_end   = j * GDparams['n_batch']
            batch_X = trainX[:, j_start:j_end]
            batch_Y = trainY[:, j_start:j_end]

            [ga_W, ga_b] = ComputeGradients(batch_X, batch_Y, W, b, lamb)
            W -= GDparams['eta']*ga_W
            b -= GDparams['eta']*ga_b

        [cost_train, loss_train] = ComputeCost(trainX, trainY, W, b, lamb)
        [cost_valid, loss_valid] = ComputeCost(validationX, validationY, W, b, lamb)
        costs_train.append(cost_train)
        costs_valid.append(cost_valid)
        losses_train.append(loss_train)
        losses_valid.append(loss_valid)
        acc_train  = ComputeAccuracy(trainX, trainY, W, b)
        acc_valid  = ComputeAccuracy(validationX, validationY, W, b)

        accs_train.append(acc_train)
        accs_valid.append(acc_valid)

        #print('Train Accuracy: ' + str(acc_train) + '    Valid Accuracy: ' + str(acc_valid))
        print('Train Cost ' + str(cost_train) + '    Valid Cost: ' + str(cost_valid))

    acc_test = ComputeAccuracy(testX, testY, W, b)
    print('Test Accuracy: ' + str(acc_test))

    plotAcc(GDparams, accs_train, accs_valid)
    plotCosts(GDparams, costs_train, costs_valid)
    plotLoss(GDparams, losses_train, losses_valid)

    return W, b

def plotCosts(GDparams, costs_train, costs_valid): # Update steps
    t = np.arange(0, len(costs_train), 1)
    fig, ax = plt.subplots()
    train = ax.plot(t, costs_train, label = 'Training')
    valid = ax.plot(t, costs_valid, label = 'Validation')
    ax.legend()
    ax.set(xlabel='Update steps', ylabel='Cost')
    plt.show()

def plotLoss(GDparams, loss_train, loss_valid): # Update steps
    t = np.arange(0, len(loss_train), 1)
    fig, ax = plt.subplots()
    train = ax.plot(t, loss_train, label = 'Training')
    valid = ax.plot(t, loss_valid, label = 'Validation')
    ax.legend()
    ax.plot(t, loss_train, t, loss_valid)
    ax.set(xlabel='Update steps', ylabel='Loss')
    plt.show()

def plotAcc(GDparams, accs_train, accs_valid): # Update steps
    t = np.arange(0, len(accs_train), 1)
    fig, ax = plt.subplots()
    ax.plot(t, accs_train, label = 'Training')
    ax.plot(t, accs_valid, label = 'Validation')
    ax.legend()
    ax.set(xlabel='Update steps', ylabel='Accuracy', label = ['Traning', 'Validation'])
    plt.show()

def getData():
    np.random.seed(seed=400)

    training_batch = LoadBatch("data_batch_1")
    trainX = training_batch.get(b'data')
    trainy = training_batch.get(b'labels')

    validation_batch = LoadBatch("data_batch_2")
    validationX = validation_batch.get(b'data')
    validationy = validation_batch.get(b'labels')

    test_batch = LoadBatch("test_batch")
    testX = test_batch.get(b'data')
    testy = test_batch.get(b'labels')

    # Transpose all matrices
    trainX = trainX.transpose()
    validationX = validationX.transpose()
    testX = testX.transpose()

    # Create one hot matrix
    trainY = np.zeros((len(trainy), max(trainy)+1))
    trainY[np.arange(len(trainy)),trainy] = 1
    trainY = trainY.transpose()

    validationY = np.zeros((len(validationy), max(validationy)+1))
    validationY[np.arange(len(validationy)),validationy] = 1
    validationY = validationY.transpose()

    testY = np.zeros((len(testy), max(testy)+1))
    testY[np.arange(len(testy)),testy] = 1
    testY = testY.transpose()

    # Normalizing
    mean_X = np.mean(trainX, 1)
    std_X  = np.std(trainX, 1)

    trainX = trainX.transpose() - mean_X
    trainX = trainX / std_X
    testX = testX.transpose() - mean_X
    testX = testX / std_X
    validationX = validationX.transpose() - mean_X
    validationX = validationX / std_X

    # Transpose once again
    trainX = trainX.transpose()
    validationX = validationX.transpose()
    testX = testX.transpose()

    return trainX, trainY, validationX, validationY, testX, testY

def main():

    [trainX, trainY, validationX, validationY, testX, testY] = getData()

    d = trainX.shape[0] # Number of dimensions
    n = trainX.shape[1] # Number of images
    K = trainY.shape[0] # Number of classes

    GDparams = {'n_batch': 100, 'eta': 0.001, 'n_epochs': 20}
    lamb = 0
    W = np.random.normal(0, 0.01, (K, d))
    b = np.random.normal(0, 0.01, (K, 1))
    [gn_W, gn_b] = ComputeGradients(trainX[:, 0:100], trainY[:, 0:100], W, b, lamb)
    #[W, b] = MiniBatchGD(trainX, trainY, validationX, validationY, testX, testY, GDparams, W, b, lamb)
    #montage(W)
    """
    [ga_W, ga_b] = ComputeGradsNumSlow(trainX[:,0:100], trainY[:,0:100], W, b, lamb, 1e-5)
    [gn_W, gn_b] = ComputeGradients(trainX[:,0:100], trainY[:,0:100], W, b, lamb)
    eps = 0.000000000001
    diff_W = np.sum(abs(ga_W - gn_W)) / max(eps, (np.sum(abs(ga_W)) + np.sum(abs(gn_W))))
    diff_b = np.sum(abs(ga_b - gn_b)) / max(eps, (np.sum(abs(ga_b)) + np.sum(abs(gn_b))))
    print("------- Relative error  -------")
    print(diff_W)
    print(diff_b)
    print('*****')
    print(np.sum(ga_W))
    print(np.sum(ga_b))
    print(np.sum(gn_W))
    print(np.sum(gn_b))
    """


if __name__ == "__main__":
    main()
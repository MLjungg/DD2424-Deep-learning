import numpy as np
from matplotlib import pyplot as plt


def soft_max(s):
    """ Standard definition of the softmax function """
    return np.exp(s) / np.sum(np.exp(s), axis=0)


def LoadBatch(filename):
    """ Copied from the dataset website """
    import pickle
    with open('../Dataset/' + filename, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
        dict[b'data'] = dict[b'data'].astype(float)

    return dict


def getData(train_file, valid_file, test_file):
    # Load data
    train_data = LoadBatch(train_file)
    valid_data = LoadBatch(valid_file)
    test_data = LoadBatch(test_file)

    # Extract data
    X_train = train_data.get(b'data')
    X_valid = valid_data.get(b'data')
    X_test = test_data.get(b'data')
    Y_train = train_data.get(b'labels')
    Y_valid = valid_data.get(b'labels')
    Y_test = test_data.get(b'labels')

    # Normalize
    X_train, X_valid, X_test = \
        normalize(X_train, X_valid, X_test)

    # One hot encoding of Y
    K = len(np.unique(train_data[b'labels']))  # Number of labels, used to create one_hot_matrix
    Y_train = one_hot_matrix(Y_train, K)
    Y_valid = one_hot_matrix(Y_valid, K)
    Y_test = one_hot_matrix(Y_test, K)

    # Transpose X
    X_train = np.transpose(X_train)
    X_valid = np.transpose(X_valid)
    X_test = np.transpose(X_test)

    return X_train, Y_train, X_valid, Y_valid, X_test, Y_test


def computeAccuracy(X, y, W, b):
    P, h = evaluateClassifier(X, W, b)

    y_pred = np.argmax(P, 0)
    y = np.argmax(y, 0)

    number_of_correct = np.sum(y_pred == y)

    N = X.shape[1]
    acc = number_of_correct / N

    return acc


def lossFunction(Y, P):
    l = - Y * np.log(P)

    return l


def ComputeCost(X, Y, W1, W2, b1, b2, lamda):
    P = evaluateClassifier(X, W1, W2, b1, b2)
    l = lossFunction(Y, P)
    N = X.shape[1]

    r = (1 / N) * (np.sum(W1 ** 2) + np.sum(W2 ** 2))

    J = np.sum(l) + lamda * r

    return J


def ComputeGradsNum(X, Y, W, b, lamda, h):
    """ Converted from matlab code """
    no = W.shape[0]

    grad_W = np.zeros(W.shape)
    grad_b = np.zeros((no, 1))

    c = ComputeCost(X, Y, W, b, lamda)

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


def ComputeGradsNumSlow(X, Y, W1, W2, b1, b2, lamda, h):
    """ Converted from matlab code """
    grad_W1 = np.zeros((W1.shape[0], W1.shape[1]))
    grad_W2 = np.zeros((W2.shape[0], W2.shape[1]))
    grad_b1 = np.zeros((b1.shape[0], b1.shape[1]))
    grad_b2 = np.zeros((b2.shape[0], b2.shape[1]))

    for i in range(len(b1)):
        b1_try = np.array(b1)
        b1_try[i] -= h
        c1 = ComputeCost(X, Y, W1, W2, b1_try, b2, lamda)

        b1_try = np.array(b1)
        b1_try[i] += h
        c2 = ComputeCost(X, Y, W1, W2, b1_try, b2, lamda)

        grad_b1[i] = (c2 - c1) / (2 * h)

    for i in range(len(b2)):
        b2_try = np.array(b2)
        b2_try[i] -= h
        c1 = ComputeCost(X, Y, W1, W2, b1, b2_try, lamda)

        b2_try = np.array(b2)
        b2_try[i] += h
        c2 = ComputeCost(X, Y, W1, W2, b1, b2_try, lamda)

        grad_b2[i] = (c2 - c1) / (2 * h)

    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            W1_try = np.array(W1)
            W1_try[i, j] -= h
            c1 = ComputeCost(X, Y, W1_try, W2, b1, b2, lamda)

            W1_try = np.array(W1)
            W1_try[i, j] += h
            c2 = ComputeCost(X, Y, W1_try, W2, b1, b2, lamda)

            grad_W1[i, j] = (c2 - c1) / (2 * h)

    for i in range(W2.shape[0]):
        for j in range(W2.shape[1]):
            W2_try = np.array(W2)
            W2_try[i, j] -= h
            c1 = ComputeCost(X, Y, W1, W2_try, b1, b2, lamda)

            W2_try = np.array(W2)
            W2_try[i, j] += h
            c2 = ComputeCost(X, Y, W1, W2_try, b1, b2, lamda)

            grad_W2[i, j] = (c2 - c1) / (2 * h)

    return [grad_W1, grad_b1, grad_W2, grad_b2]


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
    mean = np.mean(train_data, 0)
    std = np.std(train_data, 0)

    train_data = train_data - mean
    train_data = train_data / std

    valid_data = valid_data - mean
    valid_data = valid_data / std

    test_data = test_data - mean
    test_data = test_data / std

    return train_data, valid_data, test_data


def save_as_mat(data, name="model"):
    """ Used to transfer a python model to matlab """
    import scipy.io as sio
    sio.savemat(name + '.mat', {name: b})


def evaluateClassifier(X, W1, W2, b1, b2):
    # Forward pass - what is the probability of each label?
    s1 = np.dot(W1, X) + b1
    h = np.maximum(0, s1)  # Transform negative values in s to zero.
    s = np.dot(W2, h) + b2
    P = soft_max(s)

    return P, h


def generate_mini_batches(X, Y, batch_size):
    # TODO: solve non-even division
    n_batches = int(X.shape[1] / batch_size)

    x_batches = np.zeros((n_batches, X.shape[0], batch_size))
    y_batches = np.zeros((n_batches, Y.shape[0], batch_size))

    for i in range(n_batches):
        x_batches[i] = X[:, i * batch_size: (i + 1) * batch_size]
        y_batches[i] = Y[:, i * batch_size: (i + 1) * batch_size]

    return x_batches, y_batches


def ComputeGrads(X, Y, W1, W2, b1, b2, lamda, batch_size):
    P, h = evaluateClassifier(X, W1, W2, b1, b2)

    G = - (Y - P)
    dl_dw1 = (1 / batch_size) * np.dot(G, np.transpose(X))
    grad_W1 = dl_dw1 + 2 * lamda * W1

    dl_dw2 = (1 / batch_size) * np.dot(G, np.transpose(h))
    grad_W2 = dl_dw2 + 2 * lamda * W2

    grad_b1 = (1 / batch_size) * np.sum(G, 1)
    grad_b2 = (1 / batch_size) * np.sum(G, 1)

    return [grad_W1, grad_b1.reshape(-1, 1), grad_W2, grad_b2.reshape(-1, 1)]

def one_hot_matrix(Y, K):
    Y_hot = np.zeros((len(Y), K))
    Y_hot[np.arange(len(Y)), Y] = 1
    Y_hot = np.transpose(Y_hot)

    return Y_hot


def plot_costs(costs_train, costs_valid):
    x = range(0, len(costs_train), 1)
    plt.plot(x, costs_train, label="Training cost")
    plt.plot(x, costs_valid, label="Validation cost")
    plt.xlabel("Epochs")
    plt.ylabel("Cost")
    plt.title("Graph comparing training and validation cost")
    plt.legend()
    plt.show()


def mini_batch_GD(X_train, Y_train, GD_params, W1, W2, b1, b2, lamda, X_valid, Y_valid):
    x_batches, y_batches = generate_mini_batches(X_train, Y_train, GD_params["batch_size"])
    costs_per_epoch_train = []
    costs_per_epoch_valid = []

    ComputeCost(x_batches[0], y_batches[0], W1, W2, b1, b2, lamda)
    for i in range(GD_params["n_epochs"]):
        for j in range(len(y_batches)):
            grad_W, grad_b = ComputeGrads(x_batches[j], y_batches[j], W1, W2, b1, b2, lamda, GD_params["batch_size"])

            W -= GD_params["eta"] * grad_W
            b -= GD_params["eta"] * grad_b
        costs_per_epoch_train.append(ComputeCost(X, Y, W, b, lamda))
        costs_per_epoch_valid.append(ComputeCost(X_valid, Y_valid, W, b, lamda))

    return W, b, costs_per_epoch_train, costs_per_epoch_valid


def compare_gradients(X_train, Y_train, W1, W2, b1, b2, batch_size, lamda):
    # W1_num, b1_num, W2_num, b2_num = ComputeGradsNumSlow(X_train[:, 0:batch_size], Y_train[:, 0:batch_size], W1, W2, b1, b2, lamda, 1e-6)
    W_anal, b_anal = ComputeGrads(X_train[:, 0:batch_size], Y_train[:, 0:batch_size], W1, W2, b1, b2, lamda,
                                  batch_size)
    eps = 1e-6
    W_diff = np.sum(abs(W_anal - W_num)) / max(eps, (np.sum(abs(W_anal)) + np.sum(abs(W_num))))
    b_diff = np.sum(abs(b_anal - b_num)) / max(eps, (np.sum(abs(b_anal)) + np.sum(abs(b_num))))
    print("W num sum: " + str(np.sum(W_num)))
    print("W anal sum: " + str(np.sum(W_anal)))
    print("b num sum: " + str(np.sum(b_num)))
    print("b anal sum: " + str(np.sum(b_anal)))
    print("Difference in W: " + str(W_diff) + "with batch size: " + str(GD_params["batch_size"]))
    print("Difference in b: " + str(b_diff) + "with batch size: " + str(GD_params["batch_size"]))


def main():
    # Load data
    X_train, Y_train, X_valid, Y_valid, X_test, Y_test = getData("data_batch_1", "data_batch_2",
                                                                 "test_batch")
    # Set seed of random generator
    # TODO: Remove when done
    np.random.seed(seed=400)

    # Initialize parameters
    m = 50
    d = np.shape(X_train)[0]  # 3072 pixels
    K = np.shape(Y_train)[0]  # number of classes / 10.
    W1 = np.random.normal(0.0, 1 / np.sqrt(d), (m, d))
    W2 = np.random.normal(0.0, 1 / np.sqrt(m), (K, m))
    b1 = np.zeros((m, 1))
    b2 = np.zeros((K, 1))
    lamda = 0
    GD_params = {"batch_size": 10, "eta": 0.001, "n_epochs": 100}

    compare_gradients(X_train, Y_train, W1, W2, b1, b2, GD_params["batch_size"], lamda)
    # Train minibatchGD
    W, b, costs_train, costs_valid = mini_batch_GD(X_train, Y_train, GD_params, W1, W2, b1, b2, lamda, X_valid, Y_valid)

    print("JENS")


main()

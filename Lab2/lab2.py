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


def getAllData():
    # Training
    batch_1 = LoadBatch("data_batch_1")
    X_1 = batch_1.get(b'data')
    Y_1 = batch_1.get(b'labels')

    batch_2 = LoadBatch("data_batch_2")
    X_2 = batch_2.get(b'data')
    Y_2 = batch_2.get(b'labels')

    batch_3 = LoadBatch("data_batch_3")
    X_3 = batch_3.get(b'data')
    Y_3 = batch_3.get(b'labels')

    batch_4 = LoadBatch("data_batch_4")
    X_4 = batch_4.get(b'data')
    Y_4 = batch_4.get(b'labels')

    batch_5 = LoadBatch("data_batch_5")
    X_5 = batch_5.get(b'data')
    Y_5 = batch_5.get(b'labels')

    # Test
    test_batch = LoadBatch("test_batch")
    X_test = test_batch.get(b'data')
    Y_test = test_batch.get(b'labels')

    # One hot encoding
    K = len(np.unique(Y_1))  # Number of labels, used to create one_hot_matrix
    Y_1 = one_hot_matrix(Y_1, K)
    Y_2 = one_hot_matrix(Y_2, K)
    Y_3 = one_hot_matrix(Y_3, K)
    Y_4 = one_hot_matrix(Y_4, K)
    Y_5 = one_hot_matrix(Y_5, K)
    Y_test = one_hot_matrix(Y_test, K)

    # Concatenate
    X_train = np.concatenate(
        (X_1[0:9000, :], X_2[0:9000, :], X_3[0:9000, :], X_4[0:9000, :], X_5[0:9000, :]), axis=0)

    X_valid = np.concatenate((X_1[9000:10000, :], X_2[9000:10000, :], X_3[9000:10000, :],
                              X_4[9000:10000, :], X_5[9000:10000, :]), axis=0)

    Y_train = np.concatenate((Y_1[:, 0:9000], Y_2[:, 0:9000], Y_3[:, 0:9000], Y_4[:, 0:9000], Y_5[:, 0:9000]), axis=1)

    Y_valid = np.concatenate((Y_1[:, 9000:10000], Y_2[:, 9000:10000], Y_3[:, 9000:10000],
                              Y_4[:, 9000:10000], Y_5[:, 9000:10000]), axis=1)

    # Normalize
    X_train, X_valid, X_test = normalize(X_train, X_valid, X_test)

    # Transpose X
    X_train = np.transpose(X_train)
    X_valid = np.transpose(X_valid)
    X_test = np.transpose(X_test)

    return X_train, X_valid, X_test, Y_train, Y_valid, Y_test


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
    P, H = evaluateClassifier(X, W1, W2, b1, b2)
    N = X.shape[1]
    l = np.sum(lossFunction(Y, P)) / N
    r = np.sum(W1 ** 2) + np.sum(W2 ** 2)

    J = l + lamda * r

    return J, l


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
    P, H = evaluateClassifier(X, W1, W2, b1, b2)

    G = - (Y - P)

    dl_dw2 = (1 / batch_size) * np.dot(G, np.transpose(H))
    grad_b2 = (1 / batch_size) * np.sum(G, 1)
    grad_W2 = dl_dw2 + 2 * lamda * W2

    G = np.dot(np.transpose(W2), G)
    G = np.multiply(G, H > 0)

    dl_dw1 = (1 / batch_size) * np.dot(G, np.transpose(X))
    grad_W1 = dl_dw1 + 2 * lamda * W1
    grad_b1 = (1 / batch_size) * np.sum(G, 1)

    return [grad_W1, grad_b1.reshape(-1, 1), grad_W2, grad_b2.reshape(-1, 1)]


def one_hot_matrix(Y, K):
    Y_hot = np.zeros((len(Y), K))
    Y_hot[np.arange(len(Y)), Y] = 1
    Y_hot = np.transpose(Y_hot)

    return Y_hot


def plot_costs(costs_train, costs_valid):
    x = range(0, len(costs_train) * 100, 1)
    plt.plot(x, costs_train, label="Training cost")
    plt.plot(x, costs_valid, label="Validation cost")
    plt.xlabel("Updates")
    plt.ylabel("Cost")
    plt.title("Graph comparing training and validation cost")
    plt.legend()
    plt.show()


def update_eta(eta_min, eta_max, n_s, t):
    l = (t // (n_s * 2))

    if (2 * l * n_s) <= t and t <= ((2 * l + 1) * n_s):
        eta = eta_min + ((t - 2 * l * n_s) * (eta_max - eta_min)) / n_s
    else:
        eta = eta_max - (((t - (2 * l + 1) * n_s) / n_s) * (eta_max - eta_min))
    return eta


def get_t(i, j, batch_size):
    return i * batch_size + j


def mini_batch_GD(X_train, Y_train, GD_params, W1, W2, b1, b2, lamda, X_valid, Y_valid):
    batch_size = GD_params["batch_size"]
    x_batches, y_batches = generate_mini_batches(X_train, Y_train, batch_size)
    costs_train = []
    costs_valid = []
    losses_train = []
    losses_valid = []
    accuracies_train = []
    accuracies_valid = []
    etas = []
    for i in range(GD_params["n_epochs"]):
        for j in range(len(x_batches)):
            eta = update_eta(GD_params["eta_min"], GD_params["eta_max"], GD_params["n_s"], get_t(i, j + 1, batch_size))
            etas.append(eta)
            grad_W1, grad_b1, grad_W2, grad_b2 = ComputeGrads(x_batches[j], y_batches[j], W1, W2, b1, b2, lamda,
                                                              GD_params["batch_size"])

            W1 -= eta * grad_W1
            W2 -= eta * grad_W2
            b1 -= eta * grad_b1
            b2 -= eta * grad_b2

        # Add cost, loss and accuracy every epoch
        #cost_train, loss_train = ComputeCost(X_train, Y_train, W1, W2, b1, b2, lamda)
        #cost_valid, loss_valid = ComputeCost(X_valid, Y_valid, W1, W2, b1, b2, lamda)
        #costs_train.append(cost_train)
        #costs_valid.append(cost_valid)
        #losses_train.append(loss_train)
        #losses_valid.append(loss_valid)
        #accuracies_train.append(ComputeAccuracy(X_train, Y_train, W1, W2, b1, b2))
        #accuracies_valid.append(ComputeAccuracy(X_valid, Y_valid, W1, W2, b1, b2))

        cycles = ((get_t(i + 1, 0, batch_size)) // (GD_params["n_s"] * 2))
        if GD_params["cycles"] == cycles:
            return W1, W2, b1, b2, costs_train, costs_valid, losses_train, losses_valid, accuracies_train, accuracies_valid

    return W1, W2, b1, b2, costs_train, costs_valid, losses_train, losses_valid, accuracies_train, accuracies_valid


def compare_gradients(X_train, Y_train, W1, W2, b1, b2, batch_size, lamda):
    W1_num, b1_num, W2_num, b2_num = ComputeGradsNumSlow(X_train[:, 0:batch_size], Y_train[:, 0:batch_size], W1, W2, b1,
                                                         b2, lamda, 1e-6)
    W1_anal, b1_anal, W2_anal, b2_anal = ComputeGrads(X_train[:, 0:batch_size], Y_train[:, 0:batch_size], W1, W2, b1,
                                                      b2, lamda,
                                                      batch_size)
    eps = 1e-6
    W1_diff = np.sum(abs(W1_anal - W1_num)) / max(eps, (np.sum(abs(W1_anal)) + np.sum(abs(W1_num))))
    b1_diff = np.sum(abs(b1_anal - b1_num)) / max(eps, (np.sum(abs(b1_anal)) + np.sum(abs(b1_num))))
    print("W num sum: " + str(np.sum(W1_num)))
    print("W anal sum: " + str(np.sum(W1_anal)))
    print("b num sum: " + str(np.sum(b1_num)))
    print("b anal sum: " + str(np.sum(b1_anal)))
    print("Difference in W: " + str(W1_diff) + "with batch size: " + str(batch_size))
    print("Difference in b: " + str(b1_diff) + "with batch size: " + str(batch_size))


def plot_costs(costs_train, costs_valid, batch_size):
    x = range(0, len(costs_train) * batch_size, batch_size)
    plt.plot(x, costs_train, label="Training cost")
    plt.plot(x, costs_valid, label="Validation cost")
    plt.xlabel("Updated")
    plt.ylabel("Cost")
    plt.title("Graph comparing training and validation cost")
    plt.legend()
    plt.show()


def plot_loss(loss_train, loss_valid, batch_size):
    x = range(0, len(loss_train) * batch_size, batch_size)
    plt.plot(x, loss_train, label="Training loss")
    plt.plot(x, loss_valid, label="Validation loss")
    plt.xlabel("Updates")
    plt.ylabel("Loss")
    plt.title("Graph comparing training and validation loss")
    plt.legend()
    plt.show()


def plot_acc(acc_train, acc_valid, batch_size):
    x = range(0, len(acc_train) * batch_size, batch_size)
    plt.plot(x, acc_train, label="Training accuracy")
    plt.plot(x, acc_valid, label="Validation accuracy")
    plt.xlabel("Updates")
    plt.ylabel("Accuracy")
    plt.title("Graph comparing training and validation accuracy")
    plt.legend()
    plt.show()


def ComputeAccuracy(X, Y, W1, W2, b1, b2):
    P, H = evaluateClassifier(X, W1, W2, b1, b2)

    Y_pred = np.argmax(P, 0)
    Y = np.argmax(Y, 0)

    number_of_correct = np.sum(Y_pred == Y)

    N = X.shape[1]
    acc = number_of_correct / N

    return acc


def main():
    # Set seed of random generator
    np.random.seed(seed=400)
    # Load data
    #X_train, Y_train, X_valid, Y_valid, X_test, Y_test = getData("data_batch_1", "data_batch_2", "test_batch")

    X_train, X_valid, X_test, Y_train, Y_valid, Y_test = getAllData()

    # Initialize parameters
    m = 50
    d = np.shape(X_train)[0]  # 3072 pixels
    K = np.shape(Y_train)[0]  # number of classes / 10.
    W1 = np.random.normal(0.0, 1 / np.sqrt(d), (m, d))
    W2 = np.random.normal(0.0, 1 / np.sqrt(m), (K, m))
    b1 = np.zeros((m, 1))
    b2 = np.zeros((K, 1))
    l_min = -3
    l_max = -2
    l = l_min + (l_max - l_min) * np.random.uniform(0, 1, 8)
    lamdas = 10**l
    GD_params = {"batch_size": 100, "eta": 0.001, "n_epochs": 100, "eta_min": 1e-5, "eta_max": 0.1, "n_s": 900,
                 "cycles": 4}

    # compare_gradients(X_train, Y_train, W1, W2, b1, b2, GD_params["batch_size"], lamda)

    # Train minibatchGD
    for lamda in lamdas:
        W1, W2, b1, b2, costs_train, costs_valid, loss_train, loss_valid, acc_train, acc_valid = mini_batch_GD(X_train,
                                                                                                           Y_train,
                                                                                                           GD_params,
                                                                                                           W1, W2, b1,
                                                                                                           b2, lamda,
                                                                                                           X_valid,
                                                                                                           Y_valid)
        acc = ComputeAccuracy(X_valid, Y_valid, W1, W2, b1, b2)
        f = open("fineTune.txt", "a")
        f.write("Lamda=" + str(lamda) + " Accuracy=" + str(acc) + "\n")
        f.close()

    # Plot
    #plot_costs(costs_train, costs_valid, GD_params["batch_size"])
    #plot_loss(loss_train, loss_valid, GD_params["batch_size"])
    #plot_acc(acc_train, acc_valid, GD_params["batch_size"])
    #acc = ComputeAccuracy(X_test, Y_test, W1, W2, b1, b2)
    #print(acc)


main()

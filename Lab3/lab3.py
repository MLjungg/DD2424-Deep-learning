import numpy as np
from matplotlib import pyplot as plt


class Neural_network():
    def __init__(self, load_all_data, shapes, activations, GD_params):
        self.load_data(load_all_data)
        self.set_GD_params(GD_params)
        self.init_parameters(shapes)
        self.activations = activations

    def set_GD_params(self, GD_params):
        cycles = GD_params["cycles"]
        n_s = GD_params["n_s"]
        batch_size = GD_params["batch_size"]
        N = len(self.X_train[1])
        epochs = (n_s * 2) * cycles / (N / batch_size)

        if (epochs % 2 != 0):
            raise Exception("Epochs is not a whole number")

        GD_params["epochs"] = int(epochs)

        self.GD_params = GD_params

    def load_data(self, load_all_data):
        if load_all_data:
            self.X_train, self.Y_train, self.X_valid, self.Y_valid, self.X_test, self.Y_test = getAllData()
        else:
            self.X_train, self.Y_train, self.X_valid, self.Y_valid, self.X_test, self.Y_test = \
                getData("data_batch_1", "data_batch_2", "test_batch")

    def init_parameters(self, shapes):
        self.d = np.shape(self.X_train)[0]  # 3072 pixels
        self.K = np.shape(self.Y_train)[0]  # number of classes / 10.
        self.m = 50  # Nodes in hidden layer
        self.W = []
        self.B = []
        self.layers = len(shapes)
        self.gamma = []
        self.beta = []
        self.means_av = []
        self.vars_av = []

        sigma = 1e-3
        #1 / np.sqrt(shape[1])

        for shape in shapes:
            w = np.random.normal(0, sigma, size=shape)
            b = np.zeros(shape[0]).reshape(shape[0], 1)
            gamma = np.ones((shape[0], 1))
            beta = np.zeros((shape[0], 1))
            mean_av = np.zeros((shape[0], 1))
            var_av = np.zeros((shape[0], 1))

            self.W.append(w)
            self.B.append(b)
            self.gamma.append(gamma)
            self.beta.append(beta)
            self.means_av.append(mean_av)
            self.vars_av.append(var_av)

    def mini_batch_GD(self):
        # Unpack
        batch_size = self.GD_params["batch_size"]
        epochs = self.GD_params["epochs"]
        run_batch_norm = self.GD_params["run_batch_norm"]
        lamda = self.GD_params["lamda"]

        x_batches, y_batches = generate_mini_batches(self.X_train, self.Y_train, batch_size)
        costs_train = []
        costs_valid = []
        losses_train = []
        losses_valid = []
        accuracies_train = []
        accuracies_valid = []
        etas = []

        for i in range(epochs):
            for j in range(len(x_batches)):
                eta = self.update_eta(get_t(i, j + 1, len(x_batches)))
                etas.append(eta)
                if run_batch_norm:
                    W_grad, B_grad, gamma_grad, beta_grad = self.computeGrads(x_batches[j], y_batches[j])
                    for layer in range(self.layers):
                        self.W[layer] -= eta * W_grad[layer]
                        self.B[layer] -= eta * B_grad[layer]
                        self.gamma[layer] -= eta * gamma_grad[layer]
                        self.beta[layer] -= eta * beta_grad[layer]

                else:
                    W_grad, B_grad = self.computeGrads(x_batches[j], y_batches[j])
                    for layer in range(self.layers):
                        self.W[layer] -= eta * W_grad[layer]
                        self.B[layer] -= eta * B_grad[layer]

            cost_train, loss_train = self.compute_cost(self.X_train, self.Y_train)
            cost_valid, loss_valid = self.compute_cost(self.X_valid, self.Y_valid)
            losses_train.append(loss_train)
            losses_valid.append(loss_valid)
            accuracy_train = self.compute_accuracy(self.X_train, self.Y_train)
            accuracy_valid = self.compute_accuracy(self.X_valid, self.Y_valid)
            accuracies_train.append(accuracy_train)
            accuracies_valid.append(accuracy_valid)

        plot_acc(accuracies_train, accuracies_valid, len(x_batches))
        plot_loss(losses_train, losses_valid, len(x_batches))
        print("Accuracy valid: " + str(accuracy_valid) + "with lambda" + str(lamda))

        # Test
        self.GD_params["is_training"] = False
        self.GD_params["is_testing"] = True
        accuracy_test = self.compute_accuracy(self.X_test, self.Y_test)
        print("Accuracy test: " + str(accuracy_test))

    def compute_cost(self, X, Y):
        run_batch_norm = self.GD_params["run_batch_norm"]

        lamda = self.GD_params["lamda"]
        if run_batch_norm:
            P, _, _, _, _, _ = self.evaluateClassifier(X)
        else:
            P, H = self.evaluateClassifier(X)

        N = X.shape[1]
        l = np.sum(lossFunction(Y, P)) / N
        for w in self.W:
            r = np.sum(w ** 2)

        J = l + lamda * r

        return J, l

    def computeGrads(self, X_batch, Y_batch):
        # Unpack
        batch_size = self.GD_params["batch_size"]
        lamda = self.GD_params["lamda"]
        run_batch_norm = self.GD_params["run_batch_norm"]

        # init grads
        W_grad = np.zeros_like(self.W)
        B_grad = np.zeros_like(self.B)

        if run_batch_norm:
            # Init more grads
            gamma_grad = np.zeros_like(self.gamma)
            beta_grad = np.zeros_like(self.beta)

            # Forward
            P_batch, H_batch, S_batch, S_hat_batch, means_batch, vars_batch = self.evaluateClassifier(X_batch)

            # Backward
            # (21)
            G_batch = - (Y_batch - P_batch)

            # Calculate the gradients for the last layer = k
            k = self.layers - 1

            # (22)
            W_grad[k] = ((1 / batch_size) * np.dot(G_batch, np.transpose(H_batch[k]))) + 2 * lamda * self.W[k]
            B_grad[k] = (1 / batch_size) * np.sum(G_batch, 1)
            B_grad[k] = B_grad[k].reshape(-1, 1)

            # 23 & 24
            G_batch = np.dot(np.transpose(self.W[k]), G_batch)
            G_batch = np.multiply(G_batch, H_batch[k] > 0)

            for layer in range(k - 1, -1, -1):
                #(25)
                gamma_grad[layer] = ((1 / batch_size) * np.multiply(G_batch, S_hat_batch[layer]))
                gamma_grad[layer] = np.sum(gamma_grad[layer], 1)
                gamma_grad[layer] = gamma_grad[layer].reshape(-1, 1)
                beta_grad[layer] = (1 / batch_size) * np.sum(G_batch, 1)
                beta_grad[layer] = beta_grad[layer].reshape(-1, 1)

                # (26)
                G_batch = np.multiply(G_batch, self.gamma[layer])

                # (27)
                G_batch = self.Batch_norm_back_pass(G_batch, S_batch[layer], means_batch[layer], vars_batch[layer])

                # (28)
                W_grad[layer] = ((1/batch_size) * np.dot(G_batch, np.transpose(H_batch[layer]))) + 2 * lamda * self.W[layer]
                B_grad[layer] = (1 / batch_size) * np.sum(G_batch, 1)
                B_grad[layer] = B_grad[layer].reshape(-1, 1)

                # (29 & 30)
                if layer > 0:
                    G_batch = np.dot(np.transpose(self.W[layer]), G_batch)
                    G_batch = np.multiply(G_batch, H_batch[layer] > 0)

            return [W_grad, B_grad, gamma_grad, beta_grad]

        else:
            # Forward
            P_batch, H_batch = self.evaluateClassifier(X_batch)

            # Backward
            G_batch = - (Y_batch - P_batch)

            for layer in range(self.layers - 1, 0, -1):
                W_grad[layer] = ((1 / batch_size) * np.dot(G_batch, np.transpose(H_batch[layer - 1]))) + 2 * lamda * self.W[
                    layer]
                B_grad[layer] = (1 / batch_size) * np.sum(G_batch, 1)
                B_grad[layer] = B_grad[layer].reshape(-1, 1)

                G_batch = np.dot(np.transpose(self.W[layer]), G_batch)
                G_batch = np.multiply(G_batch, H_batch[layer - 1] > 0)

            W_grad[0] = (1 / batch_size) * np.dot(G_batch, np.transpose(X_batch)) + lamda * self.W[0]  # TODO: 2*lamda?
            B_grad[0] = (1 / batch_size) * np.dot(G_batch, np.ones(batch_size))
            B_grad[0] = B_grad[0].reshape(-1, 1)

            return [W_grad, B_grad]

    def Batch_norm_back_pass(self, G_batch, S_batch, mean_batch, var_batch):
        eps = self.GD_params["eps"]
        N = G_batch.shape[1]

        # (31, 32)
        sigma1 = np.power(var_batch + eps, -0.5)
        sigma2 = np.power(var_batch + eps, -1.5)

        # (33, 34)
        G1 = np.multiply(G_batch, sigma1)
        G2 = np.multiply(G_batch, sigma2)

        # (35)
        D = S_batch - mean_batch

        # (36)
        c = np.sum(np.multiply(G2, D), axis=1, keepdims=True)

        # (37)
        G_batch = G1 - 1/N * np.sum(G1, axis=1, keepdims=True) - \
                1/N * np.multiply(D, c)

        return G_batch

    def evaluateClassifier(self, X):
        run_batch_norm = self.GD_params["run_batch_norm"]
        alpha = self.GD_params["alpha"]
        is_testing = self.GD_params["is_testing"]
        is_training = self.GD_params["is_training"]
        eps = self.GD_params["eps"]


        N = X.shape[1]
        H = []
        S = []
        s = X.copy()  # s=x in first iteration

        if run_batch_norm:
            S_hat = []
            means = []
            vars = []

            for layer, (w, b, gamma, beta, mean_av, var_av, activation) in enumerate(zip(self.W, self.B, self.gamma, self.beta, self.means_av, self.vars_av, self.activations)):
                H.append(s)
                s = np.dot(w, s) + b

                # For layer = 1, 2, . . . , k − 1
                if layer < self.layers-1:
                    S.append(s)

                    if is_testing:
                        s_hat = self.batch_normalize(s, mean_av, var_av)
                    else:
                        mean = np.mean(s, axis=1, keepdims=True)
                        means.append(mean)
                        var = np.var(s, axis=1, keepdims=True) * (N-1)/N
                        vars.append(var)

                        if is_training:
                            self.means_av[layer] = alpha * mean_av + (1 - alpha) * mean
                            self.vars_av[layer] = alpha * var_av + (1 - alpha) * var

                        s_hat = self.batch_normalize(s, mean, var)

                    S_hat.append(s_hat)

                    s_tilde = np.multiply(gamma, s_hat) + beta
                    s = np.maximum(0, s_tilde)

                else:
                    P = soft_max(s)

            return P, H, S, S_hat, means, vars

        else:
            for w, b, activation in zip(self.W, self.B, self.activations):
                s = np.dot(w, s) + b
                if activation == "relu":
                    s = np.maximum(0, s)
                    H.append(s)
                if activation == "softmax":
                    P = soft_max(s)
                S.append(s)

        return P, H

    def batch_normalize(self, s, mean, var):
        eps = self.GD_params["eps"]
        s_hat = (s - mean)/np.sqrt(var + eps)

        return s_hat

    def compute_gradients_num(self, X_batch, Y_batch, size, h):
        run_batch_norm = self.GD_params["run_batch_norm"]

        grad_W = []
        grad_B = []
        for j in range(len(self.B)):
            grad_W.append(np.zeros(self.W[j].shape))
            for i in range(len(self.W[j].flatten())):
                old_par = self.W[j].flat[i]
                self.W[j].flat[i] = old_par + h
                c2, _ = self.compute_cost(X_batch, Y_batch)
                self.W[j].flat[i] = old_par - h
                c3, _ = self.compute_cost(X_batch, Y_batch)
                self.W[j].flat[i] = old_par
                grad_W[j].flat[i] = (c2 - c3) / (2 * h)

        for j in range(len(self.B)):
            grad_B.append(np.zeros(self.B[j].shape))
            for i in range(len(self.B[j].flatten())):
                old_par = self.B[j].flat[i]
                self.B[j].flat[i] = old_par + h
                c2, _ = self.compute_cost(X_batch, Y_batch)
                self.B[j].flat[i] = old_par - h
                c3, _ = self.compute_cost(X_batch, Y_batch)
                self.B[j].flat[i] = old_par
                grad_B[j].flat[i] = (c2 - c3) / (2 * h)

        if run_batch_norm:
            grad_gamma = []
            grad_beta = []

            for j in range(len(self.B)):
                grad_gamma.append(np.zeros(self.gamma[j].shape))
                for i in range(len(self.gamma[j].flatten())):
                    old_par = self.gamma[j].flat[i]
                    self.gamma[j].flat[i] = old_par + h
                    c2, _ = self.compute_cost(X_batch, Y_batch)
                    self.gamma[j].flat[i] = old_par - h
                    c3, _ = self.compute_cost(X_batch, Y_batch)
                    self.gamma[j].flat[i] = old_par
                    grad_gamma[j].flat[i] = (c2 - c3) / (2 * h)

            for j in range(len(self.B)):
                grad_beta.append(np.zeros(self.beta[j].shape))
                for i in range(len(self.beta[j].flatten())):
                    old_par = self.beta[j].flat[i]
                    self.beta[j].flat[i] = old_par + h
                    c2, _ = self.compute_cost(X_batch, Y_batch)
                    self.beta[j].flat[i] = old_par - h
                    c3, _ = self.compute_cost(X_batch, Y_batch)
                    self.beta[j].flat[i] = old_par
                    grad_beta[j].flat[i] = (c2 - c3) / (2 * h)

            return grad_W, grad_B, grad_gamma, grad_beta


        return grad_W, grad_B

    def compare_gradient(self):
        batch_size = self.GD_params["batch_size"]
        run_batch_norm = self.GD_params["run_batch_norm"]

        eps = 1e-6

        if run_batch_norm:
            W_num, B_num, gamma_num, beta_num = self.compute_gradients_num(self.X_train[:100, 0:batch_size],
                                                      self.Y_train[:100, 0:batch_size], 2, eps)
            W_anal, B_anal, gamma_anal, beta_anal = self.computeGrads(self.X_train[:100, 0:batch_size], self.Y_train[:100, 0:batch_size])


        else:

            W_num, B_num = self.compute_gradients_num(self.X_train[:100, 0:batch_size], self.Y_train[:100, 0:batch_size], 2, eps)
            W_anal, B_anal = self.computeGrads(self.X_train[:100, 0:batch_size], self.Y_train[:100, 0:batch_size])

        for layer in range(self.layers):
            W_diff = np.sum(abs(W_anal[layer] - W_num[layer])) / max(eps, (
                    np.sum(abs(W_anal[layer])) + np.sum(abs(W_num[layer]))))
            b_diff = np.sum(abs(B_anal[layer] - B_num[layer])) / max(eps, (
                    np.sum(abs(B_anal[layer])) + np.sum(abs(B_num[layer]))))
            print("W num sum in layer: " + str(layer) + ": = " + str(np.sum(W_num[layer])))
            print("W anal sum in layer" + str(layer) + ": = " + str(np.sum(W_anal[layer])))
            print("b num sum in layer" + str(layer) + ": = " + str(np.sum(B_num[layer])))
            print("b anal sum in layer " + str(layer) + ": = " + str(np.sum(B_anal[layer])))
            print("Difference in W in layer " + str(layer) + ": = " + str(W_diff) + " with batch size: " + str(
                batch_size))
            print("Difference in B in layer: " + str(layer) + ": = " + str(b_diff) + " with batch size: " + str(
                batch_size))

            if run_batch_norm:
                gamma_diff = np.sum(abs(gamma_anal[layer] - gamma_num[layer])) / max(eps, (
                        np.sum(abs(gamma_anal[layer])) + np.sum(abs(gamma_num[layer]))))
                beta_diff = np.sum(abs(beta_anal[layer] - beta_num[layer])) / max(eps, (
                        np.sum(abs(beta_anal[layer])) + np.sum(abs(beta_num[layer]))))
                print("gamma num sum in layer: " + str(layer) + ": = " + str(np.sum(gamma_num[layer])))
                print("gamma anal sum in layer" + str(layer) + ": = " + str(np.sum(gamma_anal[layer])))
                print("beta num sum in layer" + str(layer) + ": = " + str(np.sum(beta_num[layer])))
                print("beta anal sum in layer " + str(layer) + ": = " + str(np.sum(beta_anal[layer])))
                print("Difference in gamma in layer " + str(layer) + ": = " + str(gamma_diff) + " with batch size: " + str(
                    batch_size))
                print("Difference in beta in layer: " + str(layer) + ": = " + str(beta_diff) + " with batch size: " + str(
                    batch_size))

                print("-------------------")

    def update_eta(self, t):
        eta_min = self.GD_params["eta_min"]
        eta_max = self.GD_params["eta_max"]
        n_s = self.GD_params["n_s"]
        l = (t // (n_s * 2))

        if (2 * l * n_s) <= t and t <= ((2 * l + 1) * n_s):
            eta = eta_min + ((t - 2 * l * n_s) * (eta_max - eta_min)) / n_s
        else:
            eta = eta_max - (((t - (2 * l + 1) * n_s) / n_s) * (eta_max - eta_min))

        return eta

    def compute_accuracy(self, X, Y):
        run_batch_norm = self.GD_params["run_batch_norm"]

        if run_batch_norm:
            P, H, _, _, _, _ = self.evaluateClassifier(X)
        else:
            P, H = self.evaluateClassifier(X)

        Y_pred = np.argmax(P, 0)
        Y = np.argmax(Y, 0)

        number_of_correct = np.sum(Y_pred == Y)

        N = X.shape[1]
        acc = number_of_correct / N

        return acc


def get_t(i, j, batch_size):
    return i * batch_size + j


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

    return X_train, Y_train, X_valid, Y_valid, X_test, Y_test


def lossFunction(Y, P):
    l = - Y * np.log(P)

    return l


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


def generate_mini_batches(X, Y, batch_size):
    # TODO: solve non-even division
    n_batches = int(X.shape[1] / batch_size)

    x_batches = np.zeros((n_batches, X.shape[0], batch_size))
    y_batches = np.zeros((n_batches, Y.shape[0], batch_size))

    for i in range(n_batches):
        x_batches[i] = X[:, i * batch_size: (i + 1) * batch_size]
        y_batches[i] = Y[:, i * batch_size: (i + 1) * batch_size]

    return x_batches, y_batches


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


def  plot_acc(acc_train, acc_valid, batch_size):
    x = range(0, len(acc_train) * batch_size, batch_size)
    plt.plot(x, acc_train, label="Training accuracy")
    plt.plot(x, acc_valid, label="Validation accuracy")
    plt.xlabel("Updates")
    plt.ylabel("Accuracy")
    plt.title("Graph comparing training and validation accuracy")
    plt.legend()
    plt.show()


def main():
    # Set seed of random generator
    np.random.seed(seed=400)

    # Parameters for network architecture and training
    GD_params = {"batch_size": 100, "eta_min": 1e-5, "eta_max": 0.1, "n_s": 2250,
                 "cycles": 2, "lamda": 0.005, "run_batch_norm": False, "eps": 2e-52, "alpha": 0.9, "is_testing": False, "is_training": True}

    #shapes = [(50, 3072), (50, 50), (10, 50)]
    #activations = ["relu", "relu", "softmax"]
    shapes = [(50, 3072), (30, 50), (20, 30), (20, 20), (10, 20), (10, 10), (10, 10), (10, 10), (10, 10)]
    activations = ["relu", "relu", "relu", "relu", "relu", "relu", "relu", "relu", "softmax"]
    #activations = ["relu", "softmax"]
    #shapes = [(50, 100), (10, 50)]

    # Set lambdas
    l_min = -3
    l_max = -2
    l = l_min + (l_max - l_min) * np.random.uniform(0, 1, 8)
    lamdas = 10 ** l

    # Create network
    neural_network = Neural_network(load_all_data=True, shapes=shapes, activations=activations, GD_params=GD_params)
    neural_network.mini_batch_GD()
    neural_network.compare_gradient()

main()

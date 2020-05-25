import collections
import numpy as np


def read_data(file_path):
    data = open(file_path, 'r', encoding="utf8").read()
    chars = list(set(data))

    return chars


def create_mappers(data):
    char_to_ind = collections.OrderedDict()
    ind_to_char = collections.OrderedDict()
    for count, char in enumerate(data):
        char_to_ind[char] = count
        ind_to_char[count] = char

    return char_to_ind, ind_to_char


class RNN:
    @staticmethod
    def soft_max(s):
        """ Standard definition of the softmax function """
        return np.exp(s) / np.sum(np.exp(s), axis=0)

    def __init__(self, data, m=100, eta=0.1,  sequence_length=25):
        # m = Hidden states
        # k = sequence length

        self.data = data
        self.char_to_ind, self.ind_to_char = self.set_mappers()
        self.sequence_length = sequence_length
        self.K = len(data)
        self.m = m
        self.eta = eta
        self.b = np.zeros((m, 1))
        self.c = np.zeros((self.K, 1))
        self.h_init = np.zeros((self.m, 1))

        sigma = 0.01
        self.U = np.random.normal(0, sigma, (m, self.K))
        self.W = np.random.normal(0, sigma, (m, m))
        self.V = np.random.normal(0, sigma, (self.K, m))

    def get_char_from_ind(self, ind):
        return(self.ind_to_char[ind])

    def get_ind_from_char(self, char):
        return(self.char_to_ind[char])

    def set_mappers(self):
        char_to_ind = collections.OrderedDict()
        ind_to_char = collections.OrderedDict()
        for count, char in enumerate(self.data):
            char_to_ind[char] = count
            ind_to_char[count] = char

        return char_to_ind, ind_to_char

    def evaluate_classifier(self, h, x):
        a = np.dot(self.W, h) + np.dot(self.U, x) + self.b
        h = np.tanh(a)
        o = np.dot(self.V, h) + self.c
        p = RNN.soft_max(o)
        return a, h, o, p

    def synthesise(self):
        text = ""
        h = self.h_init
        dummy_index = 0
        x_one_hot = self.get_one_hot_vector(dummy_index)  # Set random x value the first iteration (X = random)

        for t in range(self.sequence_length):
            a, h, o, p = self.evaluate_classifier(h, x_one_hot)

            x_ind = get_idx_from_prob(p)
            x_one_hot = self.get_one_hot_vector(x_ind)

            char = self.get_char_from_ind(x_ind)
            text += char
        return p

    def get_one_hot_vector(self, idx):
        # If idx is one_hot_encoded
        if isinstance(idx, list):
            idx = np.where(idx == 1)

        idx_hot = np.zeros((self.K))
        idx_hot[idx] = 1

        return idx_hot

    def get_X(self):
        X_chars = self.data[0: self.sequence_length]
        X_one_hot = np.zeros((self.K, self.sequence_length))

        for idx, char in enumerate(X_chars):
            ind = self.get_ind_from_char(char)
            X_one_hot[:,idx] = self.get_one_hot_vector(ind)

        return X_one_hot

    def get_Y(self):
        Y_chars = self.data[1: self.sequence_length+1]
        Y_one_hot = np.zeros((self.K, self.sequence_length))

        for idx, char in enumerate(Y_chars):
            ind = self.get_ind_from_char(char)
            Y_one_hot[:,idx] = self.get_one_hot_vector(ind)

        return Y_one_hot

    def compute_gradients(self):
        # get data
        X = self.get_X()
        Y = self.get_Y()

        # Dictionaries for storing values during the forward pass
        aa, xx, hh, oo, pp = {}, {}, {}, {}, {}
        hh[-1] = np.copy(self.h_init)
        loss = 0

        # We're going forward!
        for t in range(self.sequence_length):
            xx[t] = X[:, [t]]

            aa[t], hh[t], oo[t], pp[t] = self.evaluate_classifier(hh[t-1], xx[t])

            loss += -np.log(np.dot(np.transpose(Y[:, t]), pp[t]))


        # We're going backward!
        grad_V, grad_W, grad_U, grad_b, grad_c = 0, 0, 0, 0, 0

        for t in range(self.sequence_length-1, -1, -1):
            grad_o = - np.transpose(Y[:, [t]] - pp[t])

            grad_V += np.dot(np.transpose(grad_o), np.transpose(hh[t]))
            grad_c += grad_o

            if t == (self.sequence_length-1):
                grad_h = np.dot(grad_o, self.V)
            else:
                grad_h = np.dot(grad_o, self.V) + np.dot(grad_a, self.W)

            grad_a = np.dot(grad_h, np.diagflat(1 - np.square(hh[t])))
            grad_W += np.dot(np.transpose(grad_a), np.transpose(hh[t-1]))
            grad_U += np.dot(np.transpose(grad_a), np.transpose(xx[t]))
            grad_b += grad_a

        """
        # Clip gradients
        grad_W = np.clip(grad_W, -5, 5)
        grad_V = np.clip(grad_V, -5, 5)
        grad_b = np.clip(grad_b, -5, 5)
        grad_U = np.clip(grad_U, -5, 5)
        grad_c = np.clip(grad_c, -5, 5)
        """
        return grad_V, grad_W, grad_U, grad_b.reshape(-1, 1), grad_c.reshape(-1, 1), loss


    def compare_gradients(self):
        grad_V_anal, grad_W_anal, grad_U_anal, grad_b_anal, grad_c_anal, loss = self.compute_gradients()
        grad_W_num, grad_U_num, grad_V_num, grad_b_num, grad_c_num = self.compute_grad_num()

        eps = 1e-6
        W_diff = np.sum(abs(grad_W_anal - grad_W_num)) / max(eps, (np.sum(abs(grad_W_anal)) + np.sum(abs(grad_W_num))))
        b_diff = np.sum(abs(grad_b_anal - grad_b_num)) / max(eps, (np.sum(abs(grad_b_anal)) + np.sum(abs(grad_b_num))))
        U_diff = np.sum(abs(grad_U_anal - grad_U_num)) / max(eps, (np.sum(abs(grad_U_anal)) + np.sum(abs(grad_U_num))))
        V_diff = np.sum(abs(grad_V_anal - grad_V_num)) / max(eps, (np.sum(abs(grad_V_anal)) + np.sum(abs(grad_V_num))))
        c_diff = np.sum(abs(grad_c_anal - grad_c_num)) / max(eps, (np.sum(abs(grad_c_anal)) + np.sum(abs(grad_c_num))))
        print("W num sum: " + str(np.sum(grad_W_num)))
        print("W anal sum: " + str(np.sum(grad_W_anal)))
        print("b num sum: " + str(np.sum(grad_b_num)))
        print("b anal sum: " + str(np.sum(grad_b_anal)))
        print("U num sum: " + str(np.sum(grad_U_num)))
        print("U anal sum: " + str(np.sum(grad_U_anal)))
        print("V num sum: " + str(np.sum(grad_V_num)))
        print("V anal sum: " + str(np.sum(grad_V_anal)))
        print("C num sum: " + str(np.sum(grad_c_num)))
        print("C anal sum: " + str(np.sum(grad_c_anal)))
        print("Difference in W: " + str(W_diff))
        print("Difference in b: " + str(b_diff))
        print("Difference in U: " + str(U_diff))
        print("Difference in V: " + str(V_diff))
        print("Difference in C: " + str(c_diff))
        print(1)


    def compute_grad_num(self, h=1e-4):
        '''Centered difference gradient'''
        # Initialize all gradients to zero
        grad_W = np.zeros(self.W.shape)
        grad_U = np.zeros(self.U.shape)
        grad_V = np.zeros(self.V.shape)
        grad_b = np.zeros(self.b.shape)
        grad_c = np.zeros(self.c.shape)

        # Gradient w.r.t W
        for j in range(self.W.shape[0]):
            for k in range(self.W.shape[1]):
                self.W[j, k] -= h
                _, _, _, _, _, c1 = self.compute_gradients()
                self.W[j, k] += 2 * h
                _, _, _, _, _, c2 = self.compute_gradients()
                self.W[j, k] -= h
                grad_W[j, k] = (c2 - c1) / (2 * h)

        # Gradient w.r.t U
        for j in range(self.U.shape[0]):
            for k in range(self.U.shape[1]):
                self.U[j, k] -= h
                _, _, _, _, _, c1 = self.compute_gradients()
                self.U[j, k] += 2 * h
                _, _, _, _, _, c2 = self.compute_gradients()
                self.U[j, k] -= h
                grad_U[j, k] = (c2 - c1) / (2 * h)

        # Gradient w.r.t V
        for j in range(self.V.shape[0]):
            for k in range(self.V.shape[1]):
                self.V[j, k] -= h
                _, _, _, _, _, c1 = self.compute_gradients()
                self.V[j, k] += 2 * h
                _, _, _, _, _, c2 = self.compute_gradients()
                self.V[j, k] -= h
                grad_V[j, k] = (c2 - c1) / (2 * h)

        # Gradient w.r.t b
        for j in range(self.b.shape[0]):
            self.b[j] -= h
            _, _, _, _, _, c1 = self.compute_gradients()
            self.b[j] += 2 * h
            _, _, _, _, _, c2 = self.compute_gradients()
            self.b[j] -= h
            grad_b[j] = (c2 - c1) / (2 * h)

        # Gradient w.r.t c
        for j in range(self.c.shape[0]):
            self.c[j] -= h
            _, _, _, _, _, c1 = self.compute_gradients()
            self.c[j] += 2 * h
            _, _, _, _, _, c2 = self.compute_gradients()
            self.c[j] -= h
            grad_c[j] = (c2 - c1) / (2 * h)

        return grad_W, grad_U, grad_V, grad_b, grad_c


def get_idx_from_prob(p):
    cp = np.cumsum(p)
    a = np.random.uniform(0, 1, 1)
    ixs = np.argwhere(cp - a > 0)
    ii = ixs[0][0]
    return ii


# Load data
book_path = "./data/goblet_book.txt"
book_chars = read_data(book_path)

# Create RNN
rnn = RNN(data=book_chars, m=25)


# Train RNN
rnn.compare_gradients()
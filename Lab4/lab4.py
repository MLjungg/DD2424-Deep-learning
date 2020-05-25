import collections
import numpy as np
from matplotlib import pyplot as plt


def read_data(file_path):
    data = open(file_path, 'r', encoding="utf8").read()

    return data


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
        self.set_data = list(set(data))
        self.char_to_ind, self.ind_to_char = self.set_mappers()
        self.sequence_length = sequence_length
        self.K = len(self.set_data)
        self.m = m
        self.eta = eta
        self.b = np.zeros((m, 1))
        self.c = np.zeros((self.K, 1))
        self.h_init = np.zeros((self.m, 1))
        self.eps = 2e-52

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
        for count, char in enumerate(self.set_data):
            char_to_ind[char] = count
            ind_to_char[count] = char

        return char_to_ind, ind_to_char

    def evaluate_classifier(self, h, x):
        a = np.dot(self.W, h) + np.dot(self.U, x) + self.b
        h = np.tanh(a)
        o = np.dot(self.V, h) + self.c
        p = RNN.soft_max(o)
        return a, h, o, p

    def synthesise(self, dummy_input, h, length_of_sequence):
        text = ""
        x_one_hot = self.get_one_hot_vector(dummy_input).reshape(self.K, 1)

        for t in range(length_of_sequence):
            a, h, o, p = self.evaluate_classifier(h, x_one_hot)

            x_ind = get_idx_from_prob(p)
            x_one_hot = self.get_one_hot_vector(x_ind).reshape(self.K, 1)

            char = self.get_char_from_ind(x_ind)
            text += char
        return text

    def get_one_hot_vector(self, idx):
        # If idx is one_hot_encoded
        if isinstance(idx, list):
            idx = np.where(idx == 1)

        idx_hot = np.zeros((self.K))
        idx_hot[idx] = 1

        return idx_hot

    def get_X(self, e):
        X_chars = self.data[e: e + self.sequence_length]
        X_one_hot = np.zeros((self.K, self.sequence_length))

        for idx, char in enumerate(X_chars):
            ind = self.get_ind_from_char(char)
            X_one_hot[:,idx] = self.get_one_hot_vector(ind)

        return X_one_hot

    def get_Y(self, e):
        Y_chars = self.data[e + 1: e + self.sequence_length+1]
        Y_one_hot = np.zeros((self.K, self.sequence_length))

        for idx, char in enumerate(Y_chars):
            ind = self.get_ind_from_char(char)
            Y_one_hot[:, idx] = self.get_one_hot_vector(ind)

        return Y_one_hot

    def compute_gradients(self, e, hprev):
        # get data
        X = self.get_X(e)
        Y = self.get_Y(e)

        # Dictionaries for storing values during the forward pass
        aa, xx, hh, oo, pp = {}, {}, {}, {}, {}
        hh[-1] = np.copy(hprev)
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


        # Clip gradients
        grad_W = np.clip(grad_W, -5, 5)
        grad_V = np.clip(grad_V, -5, 5)
        grad_b = np.clip(grad_b, -5, 5)
        grad_U = np.clip(grad_U, -5, 5)
        grad_c = np.clip(grad_c, -5, 5)

        # Store latest hidden state
        h = hh[self.sequence_length-1]

        return grad_V, grad_W, grad_U, grad_b.reshape(-1, 1), grad_c.reshape(-1, 1), loss, h


    def compare_gradients(self):
        grad_V_anal, grad_W_anal, grad_U_anal, grad_b_anal, grad_c_anal, loss, _ = self.compute_gradients(0, self.h_init)
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
        print("Difference in B: " + str(V_diff))
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
                _, _, _, _, _, c1, _ = self.compute_gradients(0, self.h_init)
                self.W[j, k] += 2 * h
                _, _, _, _, _, c2, _ = self.compute_gradients(0, self.h_init)
                self.W[j, k] -= h
                grad_W[j, k] = (c2 - c1) / (2 * h)

        # Gradient w.r.t U
        for j in range(self.U.shape[0]):
            for k in range(self.U.shape[1]):
                self.U[j, k] -= h
                _, _, _, _, _, c1, _ = self.compute_gradients(0, self.h_init)
                self.U[j, k] += 2 * h
                _, _, _, _, _, c2, _ = self.compute_gradients(0, self.h_init)
                self.U[j, k] -= h
                grad_U[j, k] = (c2 - c1) / (2 * h)

        # Gradient w.r.t V
        for j in range(self.V.shape[0]):
            for k in range(self.V.shape[1]):
                self.V[j, k] -= h
                _, _, _, _, _, c1, _ = self.compute_gradients(0, self.h_init)
                self.V[j, k] += 2 * h
                _, _, _, _, _, c2, _ = self.compute_gradients(0, self.h_init)
                self.V[j, k] -= h
                grad_V[j, k] = (c2 - c1) / (2 * h)

        # Gradient w.r.t b
        for j in range(self.b.shape[0]):
            self.b[j] -= h
            _, _, _, _, _, c1, _ = self.compute_gradients(0, self.h_init)
            self.b[j] += 2 * h
            _, _, _, _, _, c2, _ = self.compute_gradients(0, self.h_init)
            self.b[j] -= h
            grad_b[j] = (c2 - c1) / (2 * h)

        # Gradient w.r.t c
        for j in range(self.c.shape[0]):
            self.c[j] -= h
            _, _, _, _, _, c1, _ = self.compute_gradients(0, self.h_init)
            self.c[j] += 2 * h
            _, _, _, _, _, c2, _ = self.compute_gradients(0, self.h_init)
            self.c[j] -= h
            grad_c[j] = (c2 - c1) / (2 * h)

        return grad_W, grad_U, grad_V, grad_b, grad_c


def get_idx_from_prob(p):
    cp = np.cumsum(p)
    a = np.random.uniform(0, 1, 1)
    ixs = np.argwhere(cp - a > 0)
    ii = ixs[0][0]
    return ii

def plot_loss(losses):
    x = range(0, len(losses))
    plt.plot(x, losses, label="Loss")
    plt.xlabel("Updates")
    plt.ylabel("Loss")
    plt.title("Graph showcasing the loss each timestep")
    plt.legend()
    plt.show()


# Load data
book_path = "./data/goblet_book.txt"
data = read_data(book_path)

# Create RNN
rnn = RNN(data=data)

# Compare gradients
rnn.compare_gradients()

# Used to train RNN
M_grad_V = np.zeros_like(rnn.V)
M_grad_W = np.zeros_like(rnn.W)
M_grad_U = np.zeros_like(rnn.U)
M_grad_b = np.zeros_like(rnn.b)
M_grad_c = np.zeros_like(rnn.c)

# Init
epochs = 0
timestep = 0
e = 0
hprev = rnn.h_init
smooth_loss = []
check_time_step = False
check_best_model = True
best_loss = 10000

while timestep < 700000:
    if e >= len(data) - rnn.sequence_length - 1:
        hprev = rnn.h_init
        e = 0
        epochs += 1

    grad_V, grad_W, grad_U, grad_b, grad_c, loss, hprev = rnn.compute_gradients(e, hprev)

    # Adaboost
    M_grad_V += np.square(grad_V)
    M_grad_W += np.square(grad_W)
    M_grad_U += np.square(grad_U)
    M_grad_b += np.square(grad_b)
    M_grad_c += np.square(grad_c)

    rnn.V += - (rnn.eta*grad_V)/np.sqrt((M_grad_V+rnn.eps))
    rnn.W += - (rnn.eta*grad_W)/np.sqrt((M_grad_W+rnn.eps))
    rnn.U += - (rnn.eta*grad_U)/np.sqrt((M_grad_U+rnn.eps))
    rnn.b += - (rnn.eta*grad_b)/np.sqrt((M_grad_b+rnn.eps))
    rnn.c += - (rnn.eta*grad_c)/np.sqrt((M_grad_c+rnn.eps))

    # Calculate smooth loss
    if epochs == 0 and timestep == 0:
        smooth_loss.append(loss)
    else:
        smooth_loss.append(0.999*smooth_loss[-1] + 0.001 * loss)

    if check_time_step:
        if ((timestep % 10000) == 0 or timestep == 0):
            print("\nAt timestep " + str(timestep) + " we have a smooth loss of: " + str(smooth_loss[-1]))

            text = rnn.synthesise(dummy_input=0, h=hprev, length_of_sequence=200)
            print(text)

    if check_best_model:
        if best_loss - smooth_loss[-1] > 0.1 and smooth_loss[-1] < 41:
            best_loss = smooth_loss[-1]
            print("\nAt timestep " + str(timestep) + " we have a smooth loss of: " + str(smooth_loss[-1]))

            text = rnn.synthesise(dummy_input=0, h=hprev, length_of_sequence=1000)
            print(text)

    # Update parameters
    e += rnn.sequence_length
    timestep += 1

if (check_time_step):
    plot_loss(smooth_loss)



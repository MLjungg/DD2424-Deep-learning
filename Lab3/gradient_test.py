import numpy as np
from training import CrossEntropyLoss, GradientDescent
from network import Linear, ReLU, Softmax, Network, BatchNorm
from data import CIFAR10, DataLoader

class TwoLayerNetwork(Network):
    def __init__(self, optimizer, weight_decay=0):
        Network.__init__(self, weight_decay)

        self.layers = [
            Linear(10, 20, optimizer=optimizer, weight_decay=weight_decay),
            BatchNorm(20), # denna la saki till
            ReLU(),
            Linear(20, 10, optimizer=optimizer, weight_decay=weight_decay),
            Softmax(),
        ]

class ThreeLayerNetwork(Network):
    def __init__(self, optimizer, weight_decay=0):
        Network.__init__(self, weight_decay)

        self.layers = [
            Linear(10, 20, optimizer=optimizer, weight_decay=weight_decay),
            BatchNorm(20), # denna la saki till
            ReLU(),
            Linear(20, 20, optimizer=optimizer, weight_decay=weight_decay),
            BatchNorm(20), # denna la saki till
            ReLU(),
            Linear(20, 10, optimizer=optimizer, weight_decay=weight_decay),
            Softmax(),
        ]

class FourLayerNetwork(Network):
    def __init__(self, optimizer, weight_decay=0):
        Network.__init__(self, weight_decay)
        self.layers = [
            Linear(10, 20, optimizer=optimizer, weight_decay=weight_decay),
            ReLU(),
            Linear(20, 20, optimizer=optimizer, weight_decay=weight_decay),
            ReLU(),
            Linear(20, 10, optimizer=optimizer, weight_decay=weight_decay),
            ReLU(),
            Linear(10, 10, optimizer=optimizer, weight_decay=weight_decay),
            Softmax(),
        ]

def compute_gradients_num(net, criterion, inputs, Y, weight_decay):
    weights, biases = net.get_parameters()
    Gammas, betas = net.get_batchnorm_params()
    criterion = CrossEntropyLoss(network=net, weight_decay=weight_decay)
    h = 1e-6

    W_grads = []
    b_grads = []
    Gamma_grads = []
    beta_grads = []

    for i, weight in enumerate(weights):
        W_grads.append(np.zeros(weight.shape))
        for (x, y), _ in np.ndenumerate(weight):
            weights[i][x, y] = weights[i][x, y] - h
            net.set_parameters(weights.copy(), biases.copy())
            c1 = criterion(net.forward(inputs), Y)

            weights[i][x, y] = weights[i][x, y] + 2 * h
            net.set_parameters(weights.copy(), biases.copy())
            c2 = criterion(net.forward(inputs), Y)

            W_grads[i][x, y] = (c2 - c1) / (2 * h)
            weights[i][x, y] = weights[i][x, y] - h

    for i, bias in enumerate(biases):
        b_grads.append(np.zeros(bias.shape))
        for (x, y), _ in np.ndenumerate(bias):
            biases[i][x, y] = biases[i][x, y] - h
            net.set_parameters(weights.copy(), biases.copy())
            c1 = criterion(net.forward(inputs), Y)

            biases[i][x, y] = biases[i][x, y] + 2 * h
            net.set_parameters(weights.copy(), biases.copy())
            c2 = criterion(net.forward(inputs), Y)

            b_grads[i][x, y] = (c2 - c1) / (2 * h)
            biases[i][x, y] = biases[i][x, y] - h

    for i, Gamma in enumerate(Gammas):
        Gamma_grads.append(np.zeros(Gamma.shape))
        for (x, y), _ in np.ndenumerate(Gamma):
            Gammas[i][x, y] = Gammas[i][x, y] - h
            net.set_batchnorm_params(Gammas.copy(), betas.copy())
            c1 = criterion(net.forward(inputs), Y)

            Gammas[i][x, y] = Gammas[i][x, y] + 2 * h
            net.set_batchnorm_params(Gammas.copy(), betas.copy())
            c2 = criterion(net.forward(inputs), Y)

            Gamma_grads[i][x,y] = (c2 - c1) / (2 * h)
            Gammas[i][x,y] = Gammas[i][x, y] - h
    
    for i, beta in enumerate(betas):
        beta_grads.append(np.zeros(beta.shape))
        for (x, y), _ in np.ndenumerate(beta):
            betas[i][x, y] = betas[i][x, y] - h
            net.set_batchnorm_params(Gammas.copy(), betas.copy())
            c1 = criterion(net.forward(inputs), Y)

            betas[i][x, y] = betas[i][x, y] + 2 * h
            net.set_batchnorm_params(Gammas.copy(), betas.copy())
            c2 = criterion(net.forward(inputs), Y)

            beta_grads[i][x,y] = (c2 - c1) / (2 * h)
            betas[i][x,y] = betas[i][x, y] - h        


    return W_grads, b_grads, Gamma_grads, beta_grads

def compare_gradients(inputs, Y, net, weight_decay):
    # Analytical gradients
    criterion = CrossEntropyLoss(network=net, weight_decay=weight_decay)
    criterion(net.forward(inputs), Y)
    net.compute_gradients2(Y)  # last layer first
    W_grads, b_grads = net.get_gradients()
    Gamma_grads, beta_grads = net.get_batchnorm_gradients()

    # Get numerical gradients
    W_grads_num, b_grads_num, Gamma_grads_num, beta_grads_num = compute_gradients_num(net, criterion, inputs, Y, weight_decay)
    W_grads_num.reverse(), b_grads_num.reverse()
    Gamma_grads_num.reverse(), beta_grads_num.reverse()
    
    highest_err_all = 0
    highest_ga = 0
    highest_gn = 0

    print("b")

    # comparison
    for i, _ in enumerate(W_grads):
        high_err = 0
        grad = W_grads[i]
        grad_num = W_grads_num[i]
        for (x, y), _ in np.ndenumerate(grad):
            ga = grad[x, y]
            gn = grad_num[x, y]
            relative_err = np.abs(ga - gn) / max(
                np.abs(ga) + np.abs(gn), np.finfo(float).eps
            )
            if relative_err > high_err:
                high_err = relative_err
                if high_err > highest_err_all:
                    highest_err_all = high_err
                    highest_ga = ga
                    highest_gn = gn

        print("W - layer: {} \thighest err: {} \tga: {} \tgn: {}".format(i + 1, high_err, highest_ga, highest_gn))

    highest_err_all = 0
    highest_ga = 0
    highest_gn = 0

    for i, _ in enumerate(b_grads):
        high_err = 0
        grad = b_grads[i]
        grad_num = b_grads_num[i]
        for (x, y), _ in np.ndenumerate(grad):
            ga = grad[x, y]
            gn = grad_num[x, y]
            relative_err = np.abs(ga - gn) / max(
                np.abs(ga) + np.abs(gn), 1e-8
            )
            if relative_err > high_err:
                high_err = relative_err
                if high_err > highest_err_all:
                    highest_err_all = high_err
                    highest_ga = ga
                    highest_gn = gn

        print("b - layer: {} \thighest err: {} \tga: {} \tgn: {}".format(i + 1, high_err, highest_ga, highest_gn))
    
    highest_err_all = 0
    highest_ga = 0
    highest_gn = 0

    for i, _ in enumerate(beta_grads):
        high_err = 0
        grad = beta_grads[i]
        grad_num = beta_grads_num[i]
        for (x, y), _ in np.ndenumerate(grad):
            ga = grad[x, y]
            gn = grad_num[x, y]
            relative_err = np.abs(ga - gn) / max(
                np.abs(ga) + np.abs(gn), 1e-8
            )
            if relative_err > high_err:
                high_err = relative_err
                highest_ga = ga
                highest_gn = gn

        print("beta - layer: {} \thighest err: {} \tga: {} \tgn: {}".format(i + 1, high_err, highest_ga, highest_gn))
    
    highest_err_all = 0
    highest_ga = 0
    highest_gn = 0

    for i, _ in enumerate(Gamma_grads):
        high_err = 0
        grad = Gamma_grads[i]
        grad_num = Gamma_grads_num[i]
        for (x, y), _ in np.ndenumerate(grad):
            ga = grad[x, y]
            gn = grad_num[x, y]
            relative_err = np.abs(ga - gn) / max(
                np.abs(ga) + np.abs(gn), 1e-8
            )
            if relative_err > high_err:
                high_err = relative_err
                if high_err > highest_err_all:
                    highest_err_all = high_err
                    highest_ga = ga
                    highest_gn = gn

        print("b - layer: {} \thighest err: {} \tga: {} \tgn: {}".format(i + 1, high_err, highest_ga, highest_gn))
    
    
    print("\n-- Highest error: {} --".format(highest_err_all))

def gradient_test():
    train_set = CIFAR10(batches=[1])
    train_loader = DataLoader(train_set, batch_size=4)
    inputs, Y, _ = next(train_loader)
    inputs = inputs[:, :10]

    for weight_decay in [0, .001, .01, .1, 1]:
        print("\nweight_decay: {}".format(weight_decay))
        net = TwoLayerNetwork(optimizer=GradientDescent(lr=0.01), weight_decay=weight_decay)
        print("2-LayerNetwork")
        compare_gradients(inputs, Y, net, weight_decay)

        net = ThreeLayerNetwork(optimizer=GradientDescent(lr=0.01), weight_decay=weight_decay)
        print("\n3-LayerNetwork")
        compare_gradients(inputs, Y, net, weight_decay)

        net = FourLayerNetwork(optimizer=GradientDescent(lr=0.01), weight_decay=weight_decay)
        print("\n4-LayerNetwork")
        compare_gradients(inputs, Y, net, weight_decay)


gradient_test() # Is working perfectly
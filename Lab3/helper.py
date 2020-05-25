import numpy as np
from config import weight_decay

def check_gradients(inputs, Y, net):
    # Real gradients
    outputs = run_forward(inputs, net)
    run_backward(outputs, Y, net)

    # Check
    # # W0
    W0 = net[0].weights
    W0_grad = net[0].W_grad
    W0_num_grad = calc_num_gradient(W0, inputs, Y, net)
    highest_err = check_gradient_error(W0_grad, W0_num_grad)
    print(f"W0 highest error: {highest_err:.2}")

    # # b0
    b0 = net[0].bias
    b0_grad = net[0].b_grad
    b0_num_grad = calc_num_gradient(b0, inputs, Y, net)
    highest_err = check_gradient_error(b0_grad, b0_num_grad)
    print(f"b0 highest error: {highest_err:.2}")

    # W1
    W1 = net[2].weights
    W1_grad = net[2].W_grad
    W1_num_grad = calc_num_gradient(W1, inputs, Y, net)
    highest_err = check_gradient_error(W1_grad, W1_num_grad)
    print(f"W1 highest error: {highest_err:.2}")

    # b1
    b1 = net[2].bias
    b1_grad = net[2].b_grad
    b1_num_grad = calc_num_gradient(b1, inputs, Y, net)
    highest_err = check_gradient_error(b1_grad, b1_num_grad)
    print(f"b1 highest error: {highest_err:.2}")


def check_gradient_error(real_grad, num_grad):
    highest_err = 0
    for i in range(real_grad.shape[0]):
        for j in range(real_grad.shape[1]):
            ga = real_grad[i, j]
            gn = num_grad[i, j]

            relative_err = np.abs(ga - gn) / max(
                np.abs(ga) + np.abs(gn), np.finfo(float).eps
            )
            if relative_err > highest_err:
                highest_err = relative_err

    return highest_err


def calc_num_gradient(W, inputs, Y, net, h=1e-6):
    num_grad = np.zeros(W.shape)

    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W[i, j] -= h
            outputs = run_forward(inputs, net)
            _, c1 = run_loss(net, outputs, Y)

            W[i, j] += 2 * h
            outputs = run_forward(inputs, net)
            _, c2 = run_loss(net, outputs, Y)

            num_grad[i, j] = (c2 - c1) / (2 * h)
            W[i, j] -= h

    return num_grad


def run_forward(inputs, net):
    x = inputs
    for layer in net:
        x = layer.forward(x)
    outputs = x
    return outputs


def run_backward(outputs, Y, net):
    G = -(Y.T - outputs.T)
    for layer in reversed(net):
        G = layer.backward(G)


def run_loss(net, outputs, Y):
    # Loss
    weights = []
    for layer in net:
        if hasattr(layer, "weights"):
            weights.append(layer.weights)

    loss, lossl2 = cross_entropy_loss(outputs, Y, weights=weights)
    return loss, lossl2


def cross_entropy_loss(outputs, onehots, weights=[]):
    l2, loss = 0, 0
    batch_size = outputs.shape[0]

    for weight in weights:
        l2 += np.sum(weight ** 2)
    l2 = l2 * weight_decay

    for i in range(batch_size):
        loss += -np.log(onehots[i] @ outputs[i])

    scaled_loss = loss / batch_size
    return scaled_loss, scaled_loss + l2

import numpy as np 
import matplotlib.pyplot as py
import pandas as pd

data = pd.read_csv('data/train.csv')
# print(data.head())

# Data preparation
data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]

data_train = data[1000: m].T
Y_train = data_train[0]
X_train = data_train[1: n]
X_train = X_train / 255
_, m_train = X_train.shape
# print(X_train, "\n", Y_train)




# initializing parameters
def init_params():
    #<----Loaded weights and biases---->
    W1 = np.load('W1.npy', allow_pickle=True) 
    b1 = np.load('b1.npy', allow_pickle=True) 
    W2 = np.load('W2.npy', allow_pickle=True) 
    b2 = np.load('b2.npy', allow_pickle=True) 

    #<----Initial weights and biases---->
    # W1 = np.random.rand(10, 784) - 0.5
    # b1 = np.random.rand(10, 1) - 0.5
    # W2 = np.random.rand(10, 10) - 0.5
    # b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

# Rectified linear unit
def ReLU(Z):
    return np.maximum(Z, 0)

# softmax
def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def deriv_ReLU(Z):
    return  Z > 0

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y
# def one_hot(Y):
#     one_hot_Y = np.zeros((Y.size, Y.max() + 1))
#     one_hot_Y[np.arange(Y.size), Y] = 1
#     one_hot_Y = one_hot_Y.T
#     return one_hot_Y


 
def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    # m = Y.size
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return  dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    np.save('W1.npy', W1)
    np.save('b1.npy', b1)
    np.save('W2.npy', W2)
    np.save('b2.npy', b2)

    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if (i % 10 == 0):
            print('iteration: ', i)
            predictions = get_predictions(A2)
            print('Accuracy: ', get_accuracy(predictions, Y))
    return W1, b1, W2, b2

        

W1, b1, W2, b2 = gradient_descent(X_train, Y_train , 0.1, 10000)


def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)

    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation="nearest")
    plt.show
# TODO: import dependencies and write unit tests below
import pytest
from nn import NeuralNetwork
from nn import preprocess
import numpy as np
nn = NeuralNetwork([{'input_dim': 4, 'output_dim': 2, 'activation': 'sigmoid'},
                    {'input_dim': 2, 'output_dim': 4, 'activation': 'sigmoid'}],
                   .5, 1, 2, 2, 'binary cross entropy')
x_test = [[2, 2, 4, 2], [3, 5, 1, 3], [5, 5, 3, 5], [2, 3, 2, 4]]
y_test = [[0.0001], [1], [1], [0.0001]]
x_val = [[2, 4, 6, 2], [0, 2, 2, 2], [3, 7, 3, 1], [3, 5, 1, 1]]
y_val = [[0.0001], [0.0001], [1], [1]]

def test_single_forward():
    Z, A = nn._single_forward(nn._param_dict['W1'], nn._param_dict['b1'], x_test, 'sigmoid')
    assert np.shape(A) == (4, 2)
    assert np.shape(Z) == (4, 2)

def test_forward():
    #A, cache = nn.forward(x_test)
    #print(A)
    pass
    A, cache = nn.forward(x_test)
    assert np.shape(A) == (4, 4)
    assert len(cache.keys()) == 5

def test_single_backprop():
    A, cache = nn.forward(x_test)
    y_hat = nn.predict(A)
    dA_curr = nn._binary_cross_entropy_backprop(x_test, y_hat)
    A_prev = cache['A2']
    Z_curr = cache['Z2']
    W_curr = nn._param_dict['W2']
    dA_prev, dW, db = nn._single_backprop(W_curr, Z_curr, A_prev,dA_curr, 'sigmoid')
    assert np.shape(dA_prev) == (4, 2)
    assert np.shape(dW) == (4, 4)
    assert np.shape(db) == (4, 1)

def test_backprop():
    A, cache = nn.forward(x_test)
    y_hat = nn.predict(A)
    grad_dict = nn.backprop(x_test, y_hat, cache)
    pass

def test_predict():
    A, cache = nn.forward(x_test)
    y_hat = nn.predict(A)
    assert np.shape(y_hat) == np.shape(x_test)

def test_binary_cross_entropy():
    A, cache = nn.forward(x_test)
    y_hat = nn.predict(A)
    x = nn._binary_cross_entropy(x_test, y_hat)
    assert np.shape(x) == ()

def test_binary_cross_entropy_backprop():
    A, cache = nn.forward(x_test)
    y_hat = nn.predict(A)
    x = nn._binary_cross_entropy_backprop(x_test, y_hat)
    assert np.shape(x) == (4, 4)

def test_mean_squared_error():
    A, cache = nn.forward(x_test)
    y_hat = nn.predict(A)
    x = nn._mean_squared_error(x_test, y_hat)
    assert np.shape(x) == ()

def test_mean_squared_error_backprop():
    pass
    A, cache = nn.forward(x_test)
    y_hat = nn.predict(A)
    x = nn._mean_squared_error_backprop(x_test, y_hat)
    assert np.shape(x) == (4, 4)

def test_sample_seqs():
    seqs = ['ATG', 'TAA', "CTG"]
    labels = ['Start', 'Stop', 'Stop']
    x, y = preprocess.sample_seqs(seqs, labels)
    assert len(x) == len(seqs)
    assert len(y) == len(labels)


def test_one_hot_encode_seqs():
    seq = 'ATCGF'
    x = preprocess.one_hot_encode_seqs(seq)
    assert x == [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    pass
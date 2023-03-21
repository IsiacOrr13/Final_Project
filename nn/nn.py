# Imports
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Union
from numpy.typing import ArrayLike


class NeuralNetwork:
    """
    This is a class that generates a fully-connected neural network.

    Parameters:
        nn_arch: List[Dict[str, float]]
            A list of dictionaries describing the layers of the neural network.
            e.g. [{'input_dim': 64, 'output_dim': 32, 'activation': 'relu'}, {'input_dim': 32, 'output_dim': 8, 'activation:': 'sigmoid'}]
            will generate a two-layer deep network with an input dimension of 64, a 32 dimension hidden layer, and an 8 dimensional output.
        lr: float
            Learning rate (alpha).
        seed: int
            Random seed to ensure reproducibility.
        batch_size: int
            Size of mini-batches used for training.
        epochs: int
            Max number of epochs for training.
        loss_function: str
            Name of loss function.

    Attributes:
        arch: list of dicts
            (see nn_arch above)
    """

    def __init__(
        self,
        nn_arch: List[Dict[str, Union[int, str]]],
        lr: float,
        seed: int,
        batch_size: int,
        epochs: int,
        loss_function: str
    ):

        # Save architecture
        self.arch = nn_arch

        # Save hyperparameters
        self._lr = lr
        self._seed = seed
        self._epochs = epochs
        self._loss_func = loss_function
        self._batch_size = batch_size

        # Initialize the parameter dictionary for use in training
        self._param_dict = self._init_params()

    def _init_params(self) -> Dict[str, ArrayLike]:
        """
        DO NOT MODIFY THIS METHOD! IT IS ALREADY COMPLETE!

        This method generates the parameter matrices for all layers of
        the neural network. This function returns the param_dict after
        initialization.

        Returns:
            param_dict: Dict[str, ArrayLike]
                Dictionary of parameters in neural network.
        """

        # Seed NumPy
        np.random.seed(self._seed)

        # Define parameter dictionary
        param_dict = {}

        # Initialize each layer's weight matrices (W) and bias matrices (b)
        for idx, layer in enumerate(self.arch):
            layer_idx = idx + 1
            input_dim = layer['input_dim']
            output_dim = layer['output_dim']
            param_dict['W' + str(layer_idx)] = np.random.randn(output_dim, input_dim) * 0.1
            param_dict['b' + str(layer_idx)] = np.random.randn(output_dim, 1) * 0.1

        return param_dict

    def _single_forward(
        self,
        W_curr: ArrayLike,
        b_curr: ArrayLike,
        A_prev: ArrayLike,
        activation: str
    ) -> Tuple[ArrayLike, ArrayLike]:
        """
        This method is used for a single forward pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            activation: str
                Name of activation function for current layer.

        Returns:
            A_curr: ArrayLike -- This is what comes out of the activation function (io)
                Current layer activation matrix.
            Z_curr: ArrayLike -- This is what goes into the activation function (io)
                Current layer linear transformed matrix.
        """
        W_curr = np.transpose(W_curr)
        b_curr = np.transpose(b_curr)
        Z_curr = np.dot(A_prev, W_curr)
        Z_curr = np.add(Z_curr, b_curr)
        if activation == 'sigmoid':
            A_curr = self._sigmoid(Z_curr)
        elif activation == 'relu':
            A_curr = self._relu(Z_curr)
        else:
            raise ValueError('Must enter valid activation function : sigmoid or relu')

        return Z_curr, A_curr

    def forward(self, X: ArrayLike) -> Tuple[ArrayLike, Dict[str, ArrayLike]]:
        """
        This method is responsible for one forward pass of the entire neural network.

        Args:
            X: ArrayLike
                Input matrix with shape [batch_size, features].

        Returns:
            output: ArrayLike
                Output of forward pass.
            cache: Dict[str, ArrayLike]:
                Dictionary storing Z and A matrices from `_single_forward` for use in backprop.
        """
        cache = {}
        A_curr= X
        cache['A0'] = X
        for index, x in enumerate(self.arch):
            A_prev = A_curr
            index += 1
            Z_curr, A_curr = self._single_forward(self._param_dict['W'+str(index)],
                                 self._param_dict['b'+str(index)],
                                 A_prev, x['activation'])
            cache['A'+str(index)] = A_curr
            cache['Z'+str(index)] = Z_curr
        return A_curr, cache


    def _single_backprop(
        self,
        W_curr: ArrayLike,
        Z_curr: ArrayLike,
        A_prev: ArrayLike,
        dA_curr: ArrayLike,
        activation_curr: str
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        This method is used for a single backprop pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            Z_curr: ArrayLike
                Current layer linear transform matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            dA_curr: ArrayLike
                Partial derivative of loss function with respect to current layer activation matrix.
            activation_curr: str
                Name of activation function of layer.

        Returns:
            dA_prev: ArrayLike
                Partial derivative of loss function with respect to previous layer activation matrix.
            dW_curr: ArrayLike
                Partial derivative of loss function with respect to current layer weight matrix.
            db_curr: ArrayLike
                Partial derivative of loss function with respect to current layer bias matrix.
        """

        if activation_curr == 'sigmoid':
            dZ_curr = self._sigmoid_backprop(dA_curr, Z_curr)
        elif activation_curr == 'relu':
            dZ_curr = self._relu_backprop(dA_curr, Z_curr)
        else:
            raise ValueError("Must enter valid activation function: sigmoid or relu")

        m = np.shape(A_prev)[1]
        dW = np.dot(np.transpose(Z_curr), A_prev) / m
        #print('dZ_curr shape')
        #print(np.shape(dZ_curr))
        db = np.sum(dZ_curr, axis=1, keepdims=True) / m
        #print('db shape')
        #print(np.shape(db))
        dA_prev = np.dot(dZ_curr, W_curr)

        return dA_prev, dW, db


    def backprop(self, y: ArrayLike, y_hat: ArrayLike, cache: Dict[str, ArrayLike]):
        """
        This method is responsible for the backprop of the whole fully connected neural network.

        Args:
            y (array-like):
                Ground truth labels.
            y_hat: ArrayLike
                Predicted output values.
            cache: Dict[str, ArrayLike]
                Dictionary containing the information about the
                most recent forward pass, specifically A and Z matrices.

        Returns:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from this pass of backprop.
        """
        if self._loss_func == 'binary cross entropy':
            dA_prev = self._binary_cross_entropy_backprop(y, y_hat)
        else:
            dA_prev = self._mean_squared_error_backprop(y, y_hat)
        grad_dict = {}
        for idx, layer in reversed(list(enumerate(self.arch))):
            idx_curr = idx + 1
            activation = layer['activation']
            dA_curr = dA_prev
            if idx == 0:
                A_prev = cache['A0']
            else:
                A_prev = cache['A' + str(idx)]
            Z_curr = cache['Z'+ str(idx_curr)]
            W_curr = self._param_dict['W' + str(idx_curr)]

            dA_prev, dW, db = self._single_backprop(W_curr, Z_curr, A_prev, dA_curr, activation)

            grad_dict['dW' + str(idx_curr)] = dW
            #grad_dict['db' + str(idx_curr)] = db
        return grad_dict

    def _update_params(self, grad_dict: Dict[str, ArrayLike]):
        """
        This function updates the parameters in the neural network after backprop. This function
        only modifies internal attributes and does not return anything

        Args:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from most recent round of backprop.
        """
        for idx, layer in enumerate(self.arch):
            idx += 1
            W = self._param_dict['W' + str(idx)]
            dW = grad_dict['dW' + str(idx)]
            b = self._param_dict['b' + str(idx)]
            #db = grad_dict['db' + str(idx)]
            self._param_dict['W' + str(idx)] = W - (dW * self._lr)
            #print('Update b shape')
            #print(np.shape(b))
            #self._param_dict['b' + str(idx)] = b - (db)

    def fit(
        self,
        X_train: ArrayLike,
        y_train: ArrayLike,
        X_val: ArrayLike,
        y_val: ArrayLike
    ) -> Tuple[List[float], List[float]]:
        """
        This function trains the neural network by backpropagation for the number of epochs defined at
        the initialization of this class instance.

        Args:
            X_train: ArrayLike
                Input features of training set.
            y_train: ArrayLike
                Labels for training set.
            X_val: ArrayLike
                Input features of validation set.
            y_val: ArrayLike
                Labels for validation set.

        Returns:
            per_epoch_loss_train: List[float]
                List of per epoch loss for training set.
            per_epoch_loss_val: List[float]
                List of per epoch loss for validation set.
        """
        train_error = []
        val_error = []
        epochs = []
        for i in range(self._epochs):
            epochs.append(i+1)
            A_train, cache_t = self.forward(X_train)
            A_val, cache_v = self.forward(X_val)
            if self._loss_func == 'binary cross entropy':
                train_error.append(self._binary_cross_entropy(y_train, A_train))
                val_error.append(self._binary_cross_entropy(y_val, A_val))
            else:
                train_error.append(self._mean_squared_error(y_train, A_train))
                val_error.append(self._mean_squared_error(y_val, A_val))
            grad_dict = self.backprop(y_train, A_train, cache_t)
            self._update_params(grad_dict)

        plt.plot(epochs, train_error, label = 'Training Error')
        plt.plot(epochs, val_error, label = 'Validation Error')
        plt.legend()
        plt.show()

        return train_error, val_error

    def predict(self, X: ArrayLike) -> ArrayLike:
        """
        This function returns the prediction of the neural network.

        Args:
            X: ArrayLike
                Input data for prediction.

        Returns:
            y_hat: ArrayLike
                Prediction from the model.
        """
        y_hat, cache = self.forward(X)
        return y_hat

    def _sigmoid(self, Z: ArrayLike) -> ArrayLike:
        """
        Sigmoid activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        return 1 / (1+np.exp(-Z))

    def _sigmoid_backprop(self, dA: ArrayLike, Z: ArrayLike):
        """
        Sigmoid derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        otpt = self._sigmoid(Z)
        if np.shape(dA) != np.shape(otpt):
            dA = np.transpose(dA) * otpt * (1-otpt)
        else:
            dA = dA * otpt * (1-otpt)
        return dA

    def _relu(self, Z: ArrayLike) -> ArrayLike:
        """
        ReLU activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        nl_transform = np.zeros((np.shape(Z)[0], np.shape(Z)[1]))
        for idx_1, i in enumerate(Z):
            for idx_2, j in enumerate(i):
                nl_transform[idx_1][idx_2] = max(0, j)
        return nl_transform

    def _relu_backprop(self, dA: ArrayLike, Z: ArrayLike) -> ArrayLike:
        """
        ReLU derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        drelu = np.array(dA, copy = True)
        if np.shape(drelu) != np.shape(Z):
            drelu = np.transpose(drelu)
            drelu[Z <= 0] = 0
        else:
            drelu[Z<=0] = 0
        return drelu

    def _binary_cross_entropy(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Binary cross entropy loss function.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            loss: float
                Average loss over mini-batch.
        """
        bce = 0
        for idx_1, i in enumerate(y_hat):
                y_curr = y[idx_1][0]
                y_pred = y_hat[idx_1][0]
                bce += abs((y_curr * np.log10(y_pred))+((1-y_curr)*np.log10(1-y_pred)))
        bce = bce / len(y)
        return bce

    def _binary_cross_entropy_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Binary cross entropy loss function derivative for backprop.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        dA = -(np.divide(y, y_hat) - np.divide((1-y), (1-y_hat)))
        return dA

    def _mean_squared_error(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Mean squared error loss.

        Args:
            y: ArrayLike
                Ground truth output.
            y_hat: ArrayLike
                Predicted output.

        Returns:
            loss: float
                Average loss of mini-batch.
        """
        mse = 0
        for idx, y_pred in enumerate(y_hat):
            mse += abs((y[idx][0] - y_pred[0])**2)
        mse = mse/(2*len(y))
        return mse

    def _mean_squared_error_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Mean square error loss derivative for backprop.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        m = np.shape(y_hat)[1]
        dA = (2/m) * np.transpose(y_hat - y)
        return dA
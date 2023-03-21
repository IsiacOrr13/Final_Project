from nn import NeuralNetwork
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

digits = load_digits(return_X_y=True)
data = digits[0]
target = digits[1]
X_train, X_test, y_train, y_test = train_test_split(data, data, test_size=0.15, random_state=42)



lr=.00002
seed=1
batch_size=2
epochs=10
nn = NeuralNetwork([{'input_dim': 64, 'output_dim': 16, 'activation': 'sigmoid'},
                    {'input_dim': 16, 'output_dim': 64, 'activation': 'sigmoid'}],
                   lr, seed, batch_size, epochs, 'binary cross entropy')

nn.fit(X_train, y_train, X_test, y_test)
pred = nn.predict(X_train)



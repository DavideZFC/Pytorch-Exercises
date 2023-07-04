from data_loader import get_data
from functions.train_on_data import train_on_data
from functions.plot_figure import plot_figure, simple_plot
from functions.get_probas import get_probas
import numpy as np

X_train, y_train = get_data()
idx = 7
label = y_train[idx]
print(label)

net = train_on_data(X_train, y_train)

test_sample = X_train[np.newaxis,idx:idx+1,:,:]

plot_figure(X_train, idx)

print('Initial probabilities')
print(get_probas(net.numpy_predict(test_sample)))

print('In cosa lo vuoi trasformare?')
target_label = int(input())

X = net.adversarial_attack(test_sample, target_label, iter = 100)

simple_plot(X)
print('Final probabilities')
print(get_probas(net.numpy_predict(X)))
from data_loader import get_data
from functions.train_on_data import train_on_data
from functions.plot_figure import plot_figure, simple_plot
from functions.get_probas import get_probas
import numpy as np

# export dataset
X_train, y_train = get_data()
idx = 7

# train the network on the dataset
net = train_on_data(X_train, y_train)

# extract one sample from the dataset, it will be used to perform the attack
test_sample = X_train[np.newaxis,idx:idx+1,:,:]

# plot the image that will be attacked
simple_plot(test_sample)

print('Initial probabilities')
print(get_probas(net.numpy_predict(test_sample)))

print('In which number do you want to transform it?')
target_label = int(input())

# perform the gradient attack
attacked_sample = net.adversarial_attack(np.copy(test_sample), target_label, iter = 10000)

# plot the attacked sample
simple_plot(attacked_sample)

print('The norm-2 difference between the original sample and the attacked one is '+str(np.sum((attacked_sample-test_sample)**2)))

print('Probabilities after the attack')
print(get_probas(net.numpy_predict(attacked_sample)))

import matplotlib.pyplot as plt

def plot_figure(data, idx):

    plt.imshow(data[idx,:,:])
    plt.show()

def simple_plot(X):

    plt.imshow(X[0,0,:,:])
    plt.show()
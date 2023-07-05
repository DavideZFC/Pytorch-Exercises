import matplotlib.pyplot as plt

def plot_figure(data, idx):

    plt.imshow(data[idx,:,:])
    plt.show()

def simple_plot(X):
    if len(X.shape) == 2:
        plt.imshow(X)
    else:
        plt.imshow(X[0,0,:,:])
    plt.show()
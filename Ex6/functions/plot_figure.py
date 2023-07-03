import matplotlib.pyplot as plt

def plot_figure(data, idx):

    plt.imshow(data[idx,:,:])
    plt.show()
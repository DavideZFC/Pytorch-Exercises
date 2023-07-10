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

def test_autoencoder(img, net):
    print('Showing original image')
    plt.imshow(img[0,0,:,:])
    plt.show()

    print('Showing predicted image')
    out = net.numpy_predict(img)
    plt.imshow(out[0,0,:,:])
    plt.show()
import numpy as np
import matplotlib.pyplot as plt

def local_max(v):
    u = np.copy(v)
    for i in range(len(v)):
        u[i] = i > 0 and i < len(v)-1 and v[i]>v[i-1] and v[i]>v[i+1]
    return u

def plot_traj(x,y, num=2):

    idx = np.random.choice(y.shape[0], num)

    for i in range(num):
        plt.plot(x, y[idx[i],:])

    first_traj = y[idx[0],:]
    a = np.argwhere(local_max(first_traj)).flatten()
    print(a)
    plt.vlines([x[a[0]], x[a[1]]], [[-1,-1]], [[1,1]], colors='red', label='wave length')
    plt.legend()
    plt.show()
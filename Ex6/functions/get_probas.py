import numpy as np

def get_probas(v):

    v = v[0]
    p = np.exp(v)/np.sum(np.exp(v))

    dic = {}
    for i in range(10):
        dic[i] = np.round(p[i],2)

    return dic
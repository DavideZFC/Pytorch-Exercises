import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_data():

    print('We are working to export your dataset')

    df_train = pd.read_csv('data/train.csv')

    y_train = df_train['label']
    X_train = np.array(df_train.drop('label', axis=1)).reshape(-1, 28, 28)

    # normalize between -1 and 1
    X_train = (2/255)*X_train - 1

    # To prevent errors the data has to be converted to float32
    X_train = X_train.astype('float32')

    # The target has to Integer for our loss function (https://stackoverflow.com/questions/60440292/runtimeerror-expected-scalar-type-long-but-found-float)
    y_train =  y_train.astype(int)

    '''
    index = 2
    plt.imshow(X_train[index])
    plt.show()
    print('The label of this picture is: ' + str(y_train[index]))
    '''

    print('Data loaded: training set size = {}'.format(X_train.shape))

    return X_train, y_train

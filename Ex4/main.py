import numpy as np
from classes.RNN import RNN
import matplotlib.pyplot as plt

# Definiamo la dimensione del batch, la lunghezza della sequenza e il numero di feature
batch_size = 32
seq_len = 100
input_size = 1

# Generiamo un dataset di time series sintetico
data = np.zeros((batch_size,seq_len,1))

for i in range(batch_size):
    data[i,:,0] = np.sin(np.linspace(i, i+10, seq_len))

plt.plot(data[0,:,0])
plt.show()

# Definiamo la dimensione dello stato nascosto
hidden_size = 64

# Creiamo la rete neurale
rnn = RNN(input_size, hidden_size, 1)

rnn.fit(data)
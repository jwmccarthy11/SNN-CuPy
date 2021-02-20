import time
import numpy as np
import matplotlib.pyplot as plt
from module.neuron import *
from module.synapse import *

T = 1000

neurons = LIFNeuronModule(
    100, -10, -65, 4, 10, 1
)

synapses = SynapseModule.connect(
    neurons, neurons, 0, 1, 1, 5, 0.05, 0.2
)

spike_mat = np.zeros((T, neurons.n))

output = synapses.propagate(cp.zeros(neurons.n, dtype=np.float32), 0)

a = time.time()

for t in range(1, T):
    input = cp.random.random(neurons.n, dtype=np.float32)
    input += output

    spikes = neurons.update(input)
    spike_mat[t] = spikes.get()

    output = synapses.propagate(spikes, t)

b = time.time() - a

plt.figure(figsize=(8, 40))
plt.imshow(spike_mat[:500, :], cmap='binary', interpolation=None)
plt.show()

print(b)
import cupy as cp
from module.synapse_module import SynapseModule

params = {
    'n_pre': 1000,
    'n_post': 500,
    'w_min': 0.,
    'w_max': 5.,
    'd_min': 1,
    'd_max': 5,
    'den': 0.5,
    'inh': 0.2
}

synapse = SynapseModule(**params)

import time

a = time.time()

for i in range(1000):
    input = cp.random.randint(0, 2, 1000).astype(cp.bool)
    synapse.propagate(input, i)

print(time.time() - a)
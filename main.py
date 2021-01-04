import time
import math
import tqdm as tqdm
import cupy as cp
import numpy as np
from module.synapse_module import SynapseModule

params = {
    'n_pre': 10000,
    'n_post': 500,
    'w_min': 0.,
    'w_max': 5.,
    'd_min': 1,
    'd_max': 5,
    'den': 0.5,
    'inh': 0.15
}

synapse = SynapseModule(**params)

input = cp.random.randint(0, 2, 10000).astype(cp.bool)
output = np.zeros((5, 500))

w = cp.asnumpy(synapse.w)
p = cp.asnumpy(synapse.i_pre)
i = cp.asnumpy(synapse.i_post)
d = cp.asnumpy(synapse.d)

# --- PROPAGATE ON CPU ---

a = time.time()  # cpu start

pbar = tqdm.tqdm(total = len(w), desc='Ground-truthing on cpu')
for idx, (i_, d_) in enumerate(zip(i, d)):
    x = i_
    y = (0 + d_) % 5
    if input[p[idx]]:
        output[y, x] += w[idx]
    pbar.update()

a = time.time() - a  # cpu end

# --- RUN KERNEL ---

b = time.time()  # kernel start

synapse.propagate(input, 0)

b = time.time() - b  # kernel end

# --- RESULTS ---

print('\nTime to execute on cpu:', a)
print('Time to execute kernel:', b)

output = output.flatten()
out = synapse.output.flatten()

count = 0
n = 4
for i, j in zip(output, out):
    print(i, j)
    count += math.isclose(i, j, abs_tol=10**-n)
print('Output accuracy to {} sig. digits: '.format(n), count / len(output) * 100, '%', sep='')
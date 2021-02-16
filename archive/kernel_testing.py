import time
import math
import tqdm as tqdm
import cupy as cp
import numpy as np
from module.synapse import SynapseModule

n_pre  = 10000
n_post = 5000
d_max  = 5

params = {
    'n_pre': n_pre,
    'n_post': n_post,
    'w_min': 0.,
    'w_max': 5.,
    'd_min': 1,
    'd_max': d_max,
    'den': 0.01,
    'inh': 0.15
}

synapse = SynapseModule(**params)

input = cp.random.randint(0, 2, n_pre).astype(cp.bool)
output = np.zeros((d_max, n_post))

w = cp.asnumpy(synapse.w)
p = cp.asnumpy(synapse.i_pre)
i = cp.asnumpy(synapse.i_post)
d = cp.asnumpy(synapse.d)

# --- PROPAGATE ON CPU ---

a = time.time()  # cpu start

pbar = tqdm.tqdm(total = len(w), desc='Ground-truthing on CPU')
for idx, (i_, d_) in enumerate(zip(i, d)):
    x = i_
    y = (0 + d_) % 5
    if input[p[idx]]:
        output[y, x] += w[idx]
    pbar.update()

a = time.time() - a  # cpu end

print('\nTime to execute on cpu:', a)

# --- RUN KERNEL ---

b = time.time()  # kernel start

synapse.propagate(input, 0)

b = time.time() - b  # kernel end

print('Time to execute kernel:', b)

# --- RESULTS ---

output = output.flatten()
out = synapse.output.flatten()

count = 0
n = 3
for i, j in zip(output, out):
    count += math.isclose(i, j, abs_tol=10**-n)
print('Output accuracy to {} decimal places: '.format(n), count / len(output) * 100, '%', sep='')
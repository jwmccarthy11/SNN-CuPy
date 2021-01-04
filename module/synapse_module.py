import cupy as cp
from module.neuron_module import *
from cuda.kernel import propagate_delayed
from abc import ABC, abstractmethod


class SynapseModule:

    def __init__(
        self,
        n_pre:  int,     # presynaptic neurons
        n_post: int,     # postsynaptic neurons
        w_min:  float,   # min synaptic weight
        w_max:  float,   # max synaptic weight
        d_min:  int,     # min axonal delay
        d_max:  int,     # max axonal delay
        den:    float,   # synaptic density
        inh:    float    # inhibitory neurons
    ):
        self.n_syn  = int(n_pre * n_post * den)
        self.n_inh  = int(n_pre * inh)
        self.n_post = n_post
        self.d_max  = d_max

        # global synaptic index - prevent duplicates
        i_glob = cp.random.choice(
            n_pre * n_post, self.n_syn, replace=False
        ).astype(cp.int32)

        # convert to pre-post
        self.i_pre = i_glob % n_pre
        self.i_post = i_glob // n_pre

        # synaptic weights
        self.w = cp.random.uniform(w_min, w_max, self.n_syn, dtype=cp.float32)
        neg = cp.argwhere(self.i_pre < self.n_inh)
        self.w[neg] *= -1

        # axonal delays
        self.d = cp.random.randint(d_min, d_max+1, self.n_syn, dtype=cp.int)

        # output matrix
        self.output = cp.zeros((d_max, n_post), dtype=cp.float32)

        self.kernel = propagate_delayed()

    @classmethod
    def connect(
        cls,
        pre:   NeuronModule,
        post:  NeuronModule,
        d_min: int,
        d_max: int,
        den:   float,
        inh:   float
    ):
        return cls(
            pre.n, post.n, d_min, d_max, den, inh
        )

    def propagate(self, input: cp.ndarray, t: int) -> cp.ndarray:
        block_size = 256
        num_blocks = int( (self.n_syn + 255) / 256 )

        self.kernel(
            (num_blocks,), (block_size,), (
                input, t, self.n_syn, self.n_post, self.d_max,
                self.output, self.w, self.d, self.i_pre, self.i_post
            )
        )

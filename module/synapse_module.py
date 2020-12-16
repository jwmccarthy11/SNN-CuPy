import cupy as cp
import neuron_module
from abc import ABC, abstractmethod


class SynapseModule:

    def __init__(
        self,
        n_pre:  int,     # presynaptic neurons
        n_post: int,     # postsynaptic neurons
        den:    float,   # synaptic density
        inh:    float    # inhibitory neurons
    ):
        self.n_syn = int(n_pre * n_post * den)
        self.n_inh = int(n_pre * n_post * inh)

        # compressed synapse matrix

        # global synaptic index - prevent duplicates
        i_glob = cp.random.choice(
            n_pre * n_post, self.n_syn, replace=False
        )

        # convert to pre-post
        self.i_pre = i_glob % n_pre
        self.i_post = i_glob // n_pre

        # synaptic weights
        self.w = cp.random.rand(self.n_syn)
        neg = cp.argwhere(self.i_pre > self.n_inh)
        self.w[neg] *= -1

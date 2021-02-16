import cupy as cp
from abc import ABC, abstractmethod
from cuda.kernel import lif_update


class NeuronModule(ABC):

    def __init__(self, n) -> None:
        self.n = n                                  # neurons
        self.output = cp.zeros(n, dtype=cp.bool)    # output spikes

    @abstractmethod
    def update(self, input: cp.ndarray) -> cp.ndarray:
        pass


class LIFNeuronModule(NeuronModule):

    def __init__(
        self,
        n:   int,
        thr: float,     # potential at which spike fires
        res: float,     # reset potential
        ref: int,       # refractory period after spike
        cm:  float,     # capacitance
        rm:  float,     # resistance
    ) -> None:
        super(LIFNeuronModule, self).__init__(n)

        # lif state variables
        self.v = cp.ones(n, dtype=cp.float32) * res
        self.t_ref = cp.zeros(n, dtype=cp.int8)

        # lif parameters
        self.params = {
            'thr': thr,
            'res': res,
            'ref': ref,
            'cm':  cm,
            'rm':  rm
        }

        # update cuda kernel
        self.kernel = lif_update(self.params)

    def update(self, input: cp.ndarray) -> cp.ndarray:
        self.kernel(
            input, self.v, self.t_ref,
            self.output, self.v, self.t_ref
        )

        return self.output

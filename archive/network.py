from module import NeuronModule, SynapseModule


class Network:

    def __init__(self):
        self.neurons = {}
        self.synapses = {}
        self.connections = {}

    def add_neurons(
        self, id: str, neurons: NeuronModule
    ):
        self.neurons[id] = neurons
        return self

    def connect(
        self, pre_id: str, post_id: str, params: dict
    ):
        pre_neurons = self.neurons[pre_id]
        post_neurons = self.neurons[post_id]
        synapses = SynapseModule.connect(
            pre_neurons, post_neurons, **params
        )
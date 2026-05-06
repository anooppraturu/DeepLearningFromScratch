import numpy as np
from layers import LinearLayer
from activations import Tanh, ReLU


def constructor(layer_specs):
    layers = []

    # read activation function
    if layer_specs['activation'] == 'Tanh':
        act = Tanh
    elif layer_specs['activation'] == 'ReLU':
        act = ReLU
    else:
        raise NotImplementedError
    
    # construct layers from layer_specs
    nl = len(layer_specs['layers'])
    for i, data in enumerate(layer_specs['layers']):
        # Linear Layer
        if data['type'] == 'linear':
            specs=data['specs']
            layers.append(
                LinearLayer(
                    d_in = specs['d_in'],
                    d_out = specs['d_out'],
                    init_mode = specs.get('init_mode', 'Glorot')
                )
            )
            # add activation function if not final layer
            if i < nl-1:
                layers.append(act())
        else:
            raise NotImplementedError
        
    return layers


class Sequential:
    def __init__(self, layer_specs):
        self.depth = len(layer_specs['layers'])
        self.layers = constructor(layer_specs)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad
    
    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

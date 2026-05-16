import numpy as np
from .layers import LinearLayer, DropoutLayer, ConvolutionalLayer, Flatten
from .activations import Tanh, ReLU
from .norms import BatchNorm


def constructor(layer_specs):
    layers = []

    for dat in layer_specs:
        kind = dat['type']
        if kind == 'linear':
            layers.append(
                LinearLayer(
                    d_in = dat['d_in'],
                    d_out = dat['d_out'],
                    init_mode = dat.get('init_mode', 'He')
                )
            )
        elif kind == 'tanh':
            layers.append(Tanh())
        elif kind == 'relu':
            layers.append(ReLU())
        elif kind == 'dropout':
            layers.append(DropoutLayer(dat['p_drop']))
        elif kind == 'convolutional':
            layers.append(
                ConvolutionalLayer(
                    out_channels = dat['c_out'],
                    in_channels = dat['c_in'],
                    filter_size = dat['f_size'],
                    stride = dat['stride'],
                    init_mode = dat.get('init_mode', 'He')
                )
            )
        elif kind == 'flatten':
            layers.append(Flatten())
        elif kind == 'batchnorm':
            layers.append(
                BatchNorm(dat['d'])
            )
        else:
            raise NotImplementedError
        
    return layers


class Sequential:
    def __init__(self, layer_specs):
        self.layers = constructor(layer_specs)
        self.logger = None

    def set_logger(self, logger):
        self.logger = logger

    def clear_logger(self):
        self.logger = None

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer.forward(x)

            if self.logger is not None:
                self.logger.log_forward(i, layer, x)

        return x
    
    def backward(self, grad):
        for i, layer in reversed(list(enumerate(self.layers))):
            grad = layer.backward(grad)

            if self.logger is not None:
                self.logger.log_backward(i, layer, grad)
                
        return grad
    
    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params
    
    def train(self):
        for layer in self.layers:
            if hasattr(layer, 'training'):
                layer.training = True

    def eval(self):
        for layer in self.layers:
            if hasattr(layer, 'training'):
                layer.training = False


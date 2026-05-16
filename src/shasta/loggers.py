import numpy as np

class SummaryStatsLogger:
    def __init__(self):
        self.forward_logs = []
        self.backward_logs = []

    def log_forward(self, i, layer, x):
        self.forward_logs.append({
            'layer_index': i,
            'layer_type': layer.__class__.__name__,
            'output_shape': x.shape,
            'activation_mean': np.mean(x),  # stats averaged over both batch and layer dimension
            'activation_std': np.std(x),
        })

    def log_backward(self, i, layer, grad):
        # grad summed over batch dimension to get total batch gradient
        self.backward_logs.append({
            'layer_index': i,
            'layer_type': layer.__class__.__name__,
            'grad_norm': np.linalg.norm(np.sum(grad, axis=0)),
            'grad_std': np.std(np.sum(grad, axis=0))
        })


class ForwardTensorLogger:
    def __init__(self):
        self.forward_logs = []
        self.backward_logs = []

    def log_forward(self, i, layer, x):
        self.forward_logs.append({
            'layer_index': i,
            'layer_type': layer.__class__.__name__,
            'activations': np.copy(x)
        })

    def log_backward(self, i, layer, grad):
        # grad summed over batch dimension to get total batch gradient
        self.backward_logs.append({
            'layer_index': i,
            'layer_type': layer.__class__.__name__,
            'grad_norm': np.linalg.norm(np.sum(grad, axis=0)),
            'grad_std': np.std(np.sum(grad, axis=0))
        })


class BackwardTensorLogger:
    def __init__(self):
        self.forward_logs = []
        self.backward_logs = []

    def log_forward(self, i, layer, x):
        self.forward_logs.append({
            'layer_index': i,
            'layer_type': layer.__class__.__name__,
            'output_shape': x.shape,
            'activation_mean': np.mean(x),  # stats averaged over both batch and layer dimension
            'activation_std': np.std(x),
        })

    def log_backward(self, i, layer, grad):
        # grad summed over batch dimension to get total batch gradient
        self.backward_logs.append({
            'layer_index': i,
            'layer_type': layer.__class__.__name__,
            'grad': np.sum(grad, axis=0),
        })


class TensorLogger:
    def __init__(self):
        self.forward_logs = []
        self.backward_logs = []

    def log_forward(self, i, layer, x):
        self.forward_logs.append({
            'layer_index': i,
            'layer_type': layer.__class__.__name__,
            'activations': np.copy(x)
        })

    def log_backward(self, i, layer, grad):
        # grad summed over batch dimension to get total batch gradient
        self.backward_logs.append({
            'layer_index': i,
            'layer_type': layer.__class__.__name__,
            'grad': np.sum(grad, axis=0),
        })
import numpy as np
from tqdm import tqdm

class WeightProbeDirections:
    def __init__(self, layer, eps=1e-12):
        self.W0 = layer.W.copy()

        self.d1 = np.random.randn(*layer.W.shape)
        self.d2 = np.random.randn(*layer.W.shape)
        #filter normalize
        norm_axes = self.get_filter_norm_axes(layer.W)
        self.d1 = self.filter_normalize(layer.W, self.d1, norm_axes)
        self.d2 = self.filter_normalize(layer.W, self.d2, norm_axes)

    def get_filter_norm_axes(self, W):
        if W.ndim == 2:
            # LinearLayer convention: W shape is (d_in, d_out)
            return (0,)
        elif W.ndim == 4:
            # ConvolutionalLayer convention: W shape is (C_out, C_in, F, F)
            return (1, 2, 3)

    def filter_normalize(self, W, D, norm_axes, eps=1e-12):
        w_norm = np.sum(W**2, axis=norm_axes, keepdims=True)
        d_norm = np.sum(D**2, axis=norm_axes, keepdims=True)
        return D * w_norm / (d_norm + eps)


class LandscapeProbe:
    def __init__(self, model, loss_fn):
        self.model = model
        self.loss_fn = loss_fn
        self.layers = []
        self.directions = []

        for layer in model.layers:
            if hasattr(layer, 'W'):
                self.layers.append(layer)
                self.directions.append(WeightProbeDirections(layer))

    def set_weights(self, x1, x2):
        for layer, dirs in zip(self.layers, self.directions):
            layer.W[...] = dirs.W0 + x1*dirs.d1 + x2*dirs.d2

    def reset_weights(self):
        for layer, dirs in zip(self.layers, self.directions):
            layer.W[...] = dirs.W0

    def compute_loss(self, loader):
        total_loss = 0.0
        Ntot = 0.0

        for xb, yb in loader:
            B = xb.shape[0]
            preds = self.model.forward(xb)
            loss = self.loss_fn.forward(preds, yb)
            total_loss += B*loss
            Ntot += B

        return total_loss / Ntot
    
    def probe_loss(self, x1, x2, loader):
        self.set_weights(x1, x2)
        loss = self.compute_loss(loader)
        self.reset_weights()
        return loss
    
    def evaluate_grid(self, loader, x1_vals, x2_vals):
        Z = np.zeros((len(x2_vals), len(x1_vals)))
        total = len(x1_vals) * len(x2_vals)

        with tqdm(total=total) as pbar:
            for i, x2 in enumerate(x2_vals):
                for j, x1 in enumerate(x1_vals):
                    self.set_weights(x1, x2)
                    Z[i, j] = self.compute_loss(loader)
                    pbar.update(1)

        self.reset_weights()
        return Z
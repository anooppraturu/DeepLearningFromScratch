import numpy as np

class LinearProbeDirections:
    def __init__(self, layer, eps=1e-12):
        self.W0 = layer.W.copy()

        self.d1 = np.random.randn(*layer.W.shape)
        self.d2 = np.random.randn(*layer.W.shape)
        #filter normalize
        w_norm = np.linalg.norm(layer.W, axis=0, keepdims=True)
        d1_norm = np.linalg.norm(self.d1, axis=0, keepdims=True)
        d2_norm = np.linalg.norm(self.d2, axis=0, keepdims=True)
        
        self.d1 *= w_norm/(d1_norm + eps)
        self.d2 *= w_norm/(d2_norm + eps)


class LandscapeProbe:
    def __init__(self, model, loss_fn):
        self.model = model
        self.loss_fn = loss_fn
        self.layers = []
        self.directions = []

        for layer in model.layers:
            if hasattr(layer, 'W'):
                self.layers.append(layer)
                self.directions.append(LinearProbeDirections(layer))

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
            B, _ = xb.shape
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
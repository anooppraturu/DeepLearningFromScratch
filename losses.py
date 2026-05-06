import numpy as np

class MSELoss:
    def __init__(self):
        self.pred = None
        self.target = None

    def forward(self, pred, target):
        """
        pred (B x D)
        target (B x D)
        """
        self.pred = pred
        self.target = target
        B = pred.shape[0]
        return 0.5 * np.sum(np.square(self.pred - self.target)) / B
    
    def backward(self):
        B = self.pred.shape[0]
        return (self.pred - self.target) / B
    

class XEntLoss:
    def __init__(self):
        self.probs = None
        self.target = None

    def softmax(self, z):
        """
        z (B x D)
        """
        max_z = np.max(z)
        exp_z = np.exp(z - max_z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, pred, target):
        """
        pred (B x D)
        target (B x D)
        """
        self.probs = self.softmax(pred)
        self.target = target
        B = self.probs.shape[0]
        return np.sum(-self.target*np.log(self.probs + 1e-12)) / B
    
    def backward(self):
        B = self.probs.shape[0]
        return (self.probs - self.target) / B

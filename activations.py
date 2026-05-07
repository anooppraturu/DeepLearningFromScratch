import numpy as np

class Tanh:
    def __init__(self):
        self.a = None

    def forward(self, z: np.ndarray) -> np.ndarray:
        """
        z (B, D) pre-activations
        """
        self.a = np.tanh(z)
        return self.a
    
    def backward(self, delta: np.ndarray) -> np.ndarray:
        """
        delta (B, D) derivative of loss w.r.t activation
        """
        return delta*(1 - np.square(self.a))
    
    def parameters(self):
        return []
    

class ReLU:
    def __init__(self):
        self.a = None

    def forward(self, z: np.ndarray) -> np.ndarray:
        """
        z (B, D) pre-activations
        """
        self.a = np.maximum(0, z)
        return self.a
    
    def backward(self, delta: np.ndarray) -> np.ndarray:
        """
        delta (B, D) derivative of loss w.r.t activation
        """
        return delta * (self.a > 0)
    
    def parameters(self):
        return []
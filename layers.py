import numpy as np


class LinearLayer:
    def __init__(self, d_in: int, d_out: int, init_mode: str = "Glorot"):
        self.dims = (d_in, d_out)

        #weight initialization
        if init_mode == "Glorot":
            init_size = np.sqrt(6.0/(d_in + d_out))
            self.W = np.random.uniform(low=-init_size, high=init_size, size=(d_in, d_out))
            self.b = np.zeros(d_out)
        elif init_mode == "He":
            self.W = np.random.randn((d_in, d_out)) * np.sqrt(2.0 / d_in)
            self.b = np.zeros(d_out)
        else:
            raise ValueError

        # gradients
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        self.x = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        x (B, d_in) input
        """
        self.x = x
        return x@self.W + self.b
    
    def backward(self, delta: np.ndarray) -> np.ndarray:
        """
        delta (B, d_out) error sensitivity
        """
        # compute gradients
        self.dW[...] = self.x.T @ delta
        self.db[...] = np.sum(delta, axis=0)
        # derivative wrt input activations
        dx = delta@self.W.T
        return dx
    
    def parameters(self):
        """
        expose parameter
        """
        return [
            {
                'name': 'W',
                'value': self.W,
                'grad': self.dW,
                'weight_decay': True,
            },
            {
                'name': 'b',
                'value': self.b,
                'grad': self.db,
                'weight_decay': False,
            }
        ]
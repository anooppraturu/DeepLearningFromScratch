import numpy as np
from model import Sequential

class SGDOptimizer:
    def __init__(self, model, lr0=0.1, alpha=1.0, tau=10000):
        # register parameters
        self.params = model.parameters()

        #scheduler hyperparams
        self.lr0 = lr0
        self.alpha = alpha
        self.tau = tau
        self.t = 0

    def step(self):
        self.t += 1
        lr = self.lr0 / np.power(1 + self.t/self.tau, self.alpha)

        for p in self.params:
            p['value'] -= lr*p['grad']
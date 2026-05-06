import numpy as np
from model import Sequential

class L2Regularizer:
    def __init__(self, lam):
        self.lam = lam

    def penalty(self, params):
        return self.lam*0.5*np.sum(np.square(params))
    
    def grad(self, params):
        return self.lam*params
    
class L1Regularizer:
    def __init__(self, lam):
        self.lam = lam

    def penalty(self, params):
        return self.lam*np.sum(np.abs(params))
    
    def grad(self, params):
        return self.lam*np.sign(params)
    
class NoWeightDecay:
    def __init__(self, lam=0.0):
        self.lam = lam

    def penalty(self, params):
        return 0.0
    
    def grad(self, params):
        return 0.0


class SGDOptimizer:
    def __init__(self, model, lr0=0.1, alpha=1.0, tau=10000, weight_decay = 0.0, reg_type = 'L2'):
        # register parameters
        self.params = model.parameters()

        # Weight Decay
        if reg_type == 'L2':
            self.reg = L2Regularizer(weight_decay)
        elif reg_type == 'L1':
            self.reg = L1Regularizer(weight_decay)
        elif reg_type == 'None':
            self.reg = NoWeightDecay()

        #scheduler hyperparams
        self.lr0 = lr0
        self.alpha = alpha
        self.tau = tau
        self.t = 0

    def step(self):
        self.t += 1
        lr = self.lr0 / np.power(1 + self.t/self.tau, self.alpha)

        for p in self.params:
            if p['weight_decay']:
                p['value'] -= lr * (p['grad'] + self.reg.grad(p['value']))
            else:
                p['value'] -= lr * p['grad']
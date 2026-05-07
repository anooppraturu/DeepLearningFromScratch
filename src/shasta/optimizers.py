import numpy as np
from .model import Sequential

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
    def __init__(self,
                  model, 
                  lr0=0.1, 
                  alpha=1.0, 
                  tau=10000, 
                  weight_decay = 0.0, 
                  reg_type = 'L2'
                  ):
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
            param = p['value']
            grad = p['grad']
            if p['weight_decay']:
                param -= lr * (grad + self.reg.grad(param))
            else:
                param -= lr * grad


class AdamOptimizer:
    def __init__(self,
                  model, 
                  b1=0.9, 
                  b2=0.999,
                  lr0=0.1,
                  alpha=1.0, 
                  tau=10000, 
                  weight_decay = 0.0, 
                  reg_type = 'L2'
                  ):
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

        #Adam hyperparameters and gradient buffers
        self.b1 = b1
        self.b2 = b2
        self.eps = 1e-8
        self.mt = [np.zeros_like(p['value']) for p in self.params]
        self.vt = [np.zeros_like(p['value']) for p in self.params]

    def step(self):
        self.t += 1
        bias_correction = np.sqrt(1 - np.power(self.b2, self.t))/(1 - np.power(self.b1, self.t))
        lr = bias_correction * self.lr0 / np.power(1 + self.t/self.tau, self.alpha)

        for i, p in enumerate(self.params):
            param = p['value']
            grad = p['grad']

            self.mt[i] = self.b1*self.mt[i] + (1 - self.b1)*grad
            self.vt[i] = self.b2*self.vt[i] + (1 - self.b2)*np.square(grad)

            if p['weight_decay']:
                param -= lr*(self.mt[i] / (np.sqrt(self.vt[i]) + self.eps) + self.reg.grad(param))
            else:
                param -= lr * self.mt[i] / (np.sqrt(self.vt[i]) + self.eps)
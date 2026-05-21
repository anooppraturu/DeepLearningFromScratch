import numpy as np

class BatchNorm:
    def __init__(self, D):
        # caching for backward pass
        self.mu = None
        self.var = None
        self.xhat = None
        
        #running statistics for inference
        self.running_mu = np.zeros(D)
        self.running_var = np.ones(D)
        self.momentum = 0.9
        self.training = True

        # parameters + gradients
        self.gamma = np.ones(D)
        self.beta = np.zeros(D)
        self.dgamma = np.zeros_like(self.gamma)
        self.dbeta = np.zeros_like(self.beta)

        #prevent divide by 0
        self.eps = 1e-6

    def forward(self, x):
        """
        x: (B, D)
        """
        if self.training:
            self.mu = np.mean(x, axis=0)
            self.var = np.mean(np.square(x - self.mu), axis=0)
            self.xhat = (x - self.mu)/ np.sqrt(self.var + self.eps)

            self.running_mu = self.momentum*self.running_mu + (1 - self.momentum)*self.mu
            self.running_var = self.momentum*self.running_var + (1 - self.momentum)*self.var
        else:
            self.xhat = (x - self.running_mu)/ np.sqrt(self.running_var + self.eps)
            
        return self.gamma*self.xhat + self.beta
    
    def backward(self, delta):
        """
        delta: (B, D)
        """
        B, D = delta.shape
        #compute gradients
        self.dgamma = np.sum(delta*self.xhat, axis=0)
        self.dbeta = np.sum(delta, axis=0)
    
        #backpropagate
        pre = self.gamma / (B * np.sqrt(self.var + self.eps))
        dx = pre * (B * delta - np.sum(delta, axis=0) - (self.xhat * np.sum(delta * self.xhat, axis=0)))
        return dx

    def parameters(self):
        return [
            {
                'name': 'gamma',
                'value': self.gamma,
                'grad': self.dgamma,
                'weight_decay': False,
            },
            {
                'name': 'beta',
                'value': self.beta,
                'grad': self.dbeta,
                'weight_decay': False,
            }
        ]
    


class BatchNorm2D:
    def __init__(self, C):
        # caching for backward pass
        self.mu = None
        self.var = None
        self.xhat = None
        
        #running statistics for inference
        self.running_mu = np.zeros(C)
        self.running_var = np.ones(C)
        self.momentum = 0.9
        self.training = True

        # parameters + gradients
        self.gamma = np.ones(C)
        self.beta = np.zeros(C)
        self.dgamma = np.zeros_like(self.gamma)
        self.dbeta = np.zeros_like(self.beta)

        #prevent divide by 0
        self.eps = 1e-6

    def forward(self, x):
        """
        x: (B, C, H, W)
        """
        if self.training:
            self.mu = np.mean(x, axis=(0, 2, 3))
            self.var = np.mean(np.square(x - self.mu[None, :, None, None]), axis=(0, 2, 3))
            self.xhat = (x - self.mu[None, :, None, None])/ np.sqrt(self.var[None, :, None, None] + self.eps)

            self.running_mu = self.momentum*self.running_mu + (1 - self.momentum)*self.mu
            self.running_var = self.momentum*self.running_var + (1 - self.momentum)*self.var
        else:
            self.xhat = (x - self.running_mu[None, :, None, None])/ np.sqrt(self.running_var[None, :, None, None] + self.eps)
            
        return self.gamma[None, :, None, None]*self.xhat + self.beta[None, :, None, None]
    
    def backward(self, delta):
        """
        delta: (B, C, H, W)
        """
        B, C, H, W = delta.shape
        N = B * H * W
        gamma = self.gamma[None, :, None, None]
        #compute gradients
        self.dgamma = np.sum(delta*self.xhat, axis=(0, 2, 3))
        self.dbeta = np.sum(delta, axis=(0, 2, 3))
    
        #backpropagate
        pre = self.gamma[None, :, None, None] / (N * np.sqrt(self.var[None, :, None, None] + self.eps))
        dx = pre * (
                N * delta 
                - np.sum(delta, axis=(0, 2, 3), keepdims=True) 
                - self.xhat * np.sum(delta * self.xhat, axis=(0, 2, 3), keepdims=True)
            )
        return dx

    def parameters(self):
        return [
            {
                'name': 'gamma',
                'value': self.gamma,
                'grad': self.dgamma,
                'weight_decay': False,
            },
            {
                'name': 'beta',
                'value': self.beta,
                'grad': self.dbeta,
                'weight_decay': False,
            }
        ]
    

class LayerNorm:
    def __init__(self, D):
        # caching for backward pass
        self.mu = None
        self.var = None
        self.xhat = None
        
        # parameters + gradients
        self.gamma = np.ones(D)
        self.beta = np.zeros(D)
        self.dgamma = np.zeros_like(self.gamma)
        self.dbeta = np.zeros_like(self.beta)

        #prevent divide by 0
        self.eps = 1e-6

    def forward(self, x):
        """
        x: (B, D)
        """
        self.mu = np.mean(x, axis=1, keepdims=True)
        self.var = np.mean(np.square(x - self.mu), axis=1, keepdims=True)
        self.xhat = (x - self.mu) / np.sqrt(self.var + self.eps)

        return self.gamma*self.xhat + self.beta
    
    def backward(self, delta):
        """
        delta: (B, D)
        """
        B, D = delta.shape
        # compute gradients
        self.dgamma = np.sum(delta * self.xhat, axis=0)
        self.dbeta = np.sum(delta, axis=0)
        # backpropagate
        pre = 1.0 / (D * np.sqrt(self.var + self.eps))
        dxhat = self.gamma*delta
        dx = pre * (
                D * dxhat - 
                np.sum(dxhat, axis=1, keepdims=True) - 
                self.xhat*np.sum(dxhat * self.xhat, axis=1, keepdims=True)
            )
        return dx
    
    def parameters(self):
        return [
            {
                'name': 'gamma',
                'value': self.gamma,
                'grad': self.dgamma,
                'weight_decay': False,
            },
            {
                'name': 'beta',
                'value': self.beta,
                'grad': self.dbeta,
                'weight_decay': False,
            }
        ]
    
    
class LayerNorm2D:
    def __init__(self, C):
        # caching for backward pass
        self.mu = None
        self.var = None
        self.xhat = None
        
        # parameters + gradients
        self.gamma = np.ones(C)
        self.beta = np.zeros(C)
        self.dgamma = np.zeros_like(self.gamma)
        self.dbeta = np.zeros_like(self.beta)

        #prevent divide by 0
        self.eps = 1e-6

    def forward(self, x):
        """
        x: (B, C, H, W)
        """
        self.mu = np.mean(x, axis=(1, 2, 3), keepdims=True)
        self.var = np.mean(np.square(x - self.mu), axis=(1, 2, 3), keepdims=True)
        self.xhat = (x - self.mu) / np.sqrt(self.var + self.eps)

        return self.gamma[None, :, None, None]*self.xhat + self.beta[None, :, None, None]
    
    def backward(self, delta):
        """
        delta: (B, C, H, W)
        """
        B, C, H, W = delta.shape
        N = C*H*W
        # compute gradients
        self.dgamma = np.sum(delta * self.xhat, axis=(0, 2, 3))
        self.dbeta = np.sum(delta, axis=(0, 2, 3))
        # backpropagate
        pre = 1.0 / (N * np.sqrt(self.var + self.eps))
        dxhat = delta*self.gamma[None, :, None, None]
        dx = pre * (
                N * dxhat - 
                np.sum(dxhat, axis=(1, 2, 3), keepdims=True) - 
                self.xhat*np.sum(dxhat * self.xhat, axis=(1, 2, 3), keepdims=True)
            )
        return dx
    
    def parameters(self):
        return [
            {
                'name': 'gamma',
                'value': self.gamma,
                'grad': self.dgamma,
                'weight_decay': False,
            },
            {
                'name': 'beta',
                'value': self.beta,
                'grad': self.dbeta,
                'weight_decay': False,
            }
        ]
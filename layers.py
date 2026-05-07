import numpy as np


class LinearLayer:
    def __init__(self, d_in: int, d_out: int, init_mode: str = "He"):
        self.dims = (d_in, d_out)

        #weight initialization
        if init_mode == "Glorot":
            init_size = np.sqrt(6.0/(d_in + d_out))
            self.W = np.random.uniform(low=-init_size, high=init_size, size=(d_in, d_out))
            self.b = np.zeros(d_out)
        elif init_mode == "He":
            self.W = np.random.randn(d_in, d_out) * np.sqrt(2.0 / d_in)
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
    

class ConvolutionalLayer:
    def __init__(self, out_channels, in_channels, filter_size, stride, init_mode: str = "He"):
        self.C_in = in_channels
        self.C_out = out_channels
        self.F = filter_size
        self.stride = stride
        
        #weight initialization
        fan_in = self.F*self.F*self.C_in
        fan_out = self.F*self.F*self.C_out
        if init_mode == "Glorot":
            init_size = np.sqrt(6.0/(fan_in + fan_out))
            self.W = np.random.uniform(low=-init_size, high=init_size, size=(self.C_out, self.C_in, self.F, self.F))
            self.b = np.zeros(self.C_out)
        elif init_mode == "He":
            self.W = np.random.randn(self.C_out, self.C_in, self.F, self.F) * np.sqrt(2.0 / fan_in)
            self.b = np.zeros(self.C_out)
        else:
            raise ValueError
        
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.x = None

    def forward(self, x):
        """
        x: (B, C_in, H, W) batch, in_channel, height, width
        """
        self.x = x
        B, C_in, H_in, W_in = x.shape
        assert C_in == self.C_in
        assert (H_in - self.F) % self.stride == 0
        assert (W_in - self.F) % self.stride == 0

        H_out = (H_in - self.F) // self.stride + 1
        W_out = (W_in - self.F) // self.stride + 1

        x_out = np.zeros(shape=(B, self.C_out, H_out, W_out))

        for i in range(0, H_out):
            for j in range(0, W_out):
                h0 = i*self.stride
                h1 = h0 + self.F
                w0 = j*self.stride
                w1 = w0 + self.F

                window = x[:, :, h0:h1, w0:w1]   # (B, C_in, F, F)
                                            # (B, , C_in, F, F)         ( , C_out, C_in, F, F)
                x_out[:, :, i, j] = np.sum(window[:,None,:,:,:]*self.W[None,:,:,:,:], axis=(2,3,4)) + self.b[None, :]

        return x_out
    
    def backward(self, delta):
        """
        delta: (B, C_out, H_out, W_out)
        dx: (B, C_in, H_in, W_in)
        """
        _, _, H_out, W_out = delta.shape
        self.db[...] = np.sum(delta, axis=(0,2,3))
        # zero grad
        self.dW.fill(0.0)
        dx = np.zeros_like(self.x)

        for m in range(H_out):
            for n in range(W_out):
                h0 = m*self.stride
                h1 = h0 + self.F
                w0 = n*self.stride
                w1 = w0 + self.F

                window = self.x[:, :, h0:h1, w0:w1]   # (B, C_in, F, F)
                dlocal = delta[:, :, m, n]            # (B, C_out)
                                        #(B, C_out, , , , m, n)               #(B, , C_in, F, F)
                self.dW[...] += np.sum(dlocal[:, :, None, None, None] * window[:, None, :, :, :], axis=0)

                dx[:, :, h0:h1, w0:w1] += np.sum(dlocal[:, :, None, None, None] * self.W[None, :, :, :, :], axis=1)

        return dx
    
    def parameters(self):
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
    


class Flatten:
    def __init__(self):
        self.input_shape = None

    def forward(self, x):
        self.input_shape = x.shape
        return x.reshape(self.input_shape[0], -1)
    
    def backward(self, delta):
        return delta.reshape(self.input_shape)
    
    def parameters(self):
        return []



class DropoutLayer:
    def __init__(self, p_drop):
        self.p = p_drop
        self.mask = None
        self.training = True

    def forward(self, x):
        """
        x (B x D)
        """
        if not self.training:
            return x
        
        self.mask = (np.random.rand(*x.shape) > (1 - self.p)) / (1 - self.p)
        return self.mask * x
    
    def backward(self, delta):
        """
        delta (B x D)
        """
        if not self.training:
            return delta
            
        return self.mask*delta
    
    def parameters(self):
        return []
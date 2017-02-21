import numpy as np
from cs231n.layers import *
from cs231n.layer_utils import *

class FullyConnectedConvNet(object):
    
    def __init__(self, input_dim=(3,32,32), num_filters=[32,10,5], filter_sizes=[7,5,3],
                 hidden_dims=[200,100], num_classes=10,weight_scale=1e-2,reg=0.0,
                 dtype=np.float64,use_batchnorm=False, dropout=0):
        
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}
        
        # Initializing the weights and biases for convolutional layer
        i = 1
        for nf in num_filters:
            C, H, W = input_dim[nf]
            self.params['W%d'%(i)] = weight_scale*np.random.randn(num_filters[nf], C, filter_sizes[nf], filter_sizes[nf])
            self.params['b%d'%(i)] = np.zeros(filter_sizes[nf])
        
        
        
        
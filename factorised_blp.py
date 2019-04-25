import torch
import torch.nn as nn
import torch.nn.functional as F

#Factorised bilinear pooling for the inputs
class fblp(nn.Module):
    """
    Replace each of these yourselves, this torch implementation is written to work with my framework
    options.pool_k      = K parameter from paper            (5)
    options.pool_o      = o parameter from paper            (1000)
    options.xpool_in    = Input size of x                   (1024)
    options.ypool_in    = Input size of y                   (1024)
    """
    def __init__(self, options):
        super(fblp, self).__init__()
        joint_emb_size  = options.pool_k * options.pool_o
        self.options    = options
        self.xproj      = nn.Linear(options.xpool_in, joint_emb_size)
        self.yproj      = nn.Linear(options.ypool_in, joint_emb_size)
    
    def forward(self, x, y):
        import ipdb; ipdb.set_trace()
        x   = self.xproj(x)                                         # batch, joint_emb_size
        y   = self.yproj(y)                                         # batch, joint_emb_size
        out = torch.mul(x, y)                                       # batch, joint_emb_size
        out = out.view(-1, 1, self.options.pool_o, self.options.pool_k) # batch, 1, o, k
        out = torch.squeeze(torch.sum(out, 3))                      # batch, o
        out = torch.sqrt(F.relu(out)) - torch.sqrt(F.relu(-out))    # Signed square root
        out = F.normalize(out)
        return(out)



from keras.layers import Layer, Reshape, Multiply
from keras import backend as K
from keras import initializers
from keras.activations import relu
class keras_fblp(Layer):
    """
    pool_k      = K parameter from paper                    (5)
    pool_o      = o parameter from paper                    (1000)
    x_in        = Input size of x                           (1024)
    y_in        = Input size of y                           (1024)
    """
    def __init__(self, pool_k, pool_o, x_in, y_in):
        self.reshape    = Reshape((pool_o, pool_k))
        self.ewmultiply = Multiply()
        self.pool_k     = pool_k
        self.pool_o = pool_o
        self.output_dim = pool_k
        self.x_in       = x_in
        self.y_in       = y_in
        super(keras_fblp, self).__init__()
    
    def build(self, input_shape):
        #Define the weights for our 2 fully connected layers (linear projections)
        self.x_weights  = self.add_weight(
            name        = 'x_weight',
            shape       = (self.x_in, self.pool_k*self.pool_o),
            initializer = 'uniform',
            trainable   = True
        )
        self.y_weights  = self.add_weight(
            name        = 'y_weight',
            shape       = (self.y_in, self.pool_k*self.pool_o),
            initializer = 'uniform',
            trainable   = True
        )
        super(keras_fblp, self).build(input_shape)
    
    def call(self, inputs):
        x, y= inputs
        x   = K.dot(x, self.x_weights)                              # batch, (pool_k*pool_o)
        y   = K.dot(y, self.y_weights)                              # batch, (pool_k*pool_o)
        out = self.ewmultiply([x, y])                               # batch, (pool_k*pool_o)
        out = self.reshape(out)                                     # batch, pool_k, pool_o
        out = K.sum(out, axis=2)                                    # batch, pool_o
        out = K.sqrt( relu( out ) ) - K.sqrt( relu( -out ) )        #Signed Square Root   
        out = K.l2_normalize( out )                                 # batch, pool_o
        return(out)
    
    def compute_output_shape(self, input_shape):
        return([input_shape[0][0], self.output_dim])

######## Torch
# test    = fblp(options)
# x       = torch.ones(32, 2048)
# y       = torch.ones(32, 2048)
# out     = test(x,y)
# print(out)
########
######## Keras
# test    = keras_fblp(5, 1000, 2048, 2048)
# x       = K.ones((32, 2048))
# y       = K.ones((32, 2048))
# out     = test([x,y])
# print(out)
########
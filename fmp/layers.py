#KerasFractionalMaxPooling

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer


class FractionalMaxPooling2D(Layer):
    def __init__(self, pooling_ratio, pseudo_random=False, overlapping=False,
                seed=0,data_format=None):
        super(FractionalMaxPooling2D, self).__init__()
        self.pooling_ratio = pooling_ratio
        self.pseudo_random = pseudo_random
        self.overlapping = overlapping
        self.seed = seed
        self.data_format = data_format
    def build(self, input_shape):
        
        pass
    def call(self, x):
        from tensorflow.nn import fractional_max_pool
        pr = [1., self.pooling_ratio[0], self.pooling_ratio[1], 1.]
        return fractional_max_pool(x,pooling_ratio=pr,
                                      pseudo_random = self.pseudo_random, 
                                      overlapping=self.overlapping,
                                      seed = self.seed
                                     )[0]  
    def compute_output_shape(self, input_shape):
        # return the output shape
        from tensorflow.math import floor
        
        
        if self.data_format == 'channels_first':
            rows = input_shape[2]
            cols = input_shape[3]
        elif self.data_format == 'channels_last':
            rows = input_shape[1]
            cols = input_shape[2]
        
        rows = floor(rows / self.pooling_ratio[0])
        cols = floor(cols / self.pooling_ratio[1])
        
        if self.data_format == 'channels_first':
            return (input_shape[0], input_shape[1], rows, cols)
        elif self.data_format == 'channels_last':
            return (input_shape[0], rows, cols, input_shape[3])
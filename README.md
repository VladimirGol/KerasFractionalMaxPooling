# KerasFractionalMaxPooling
An implementation of FractionalMaxPooling Layer in Keras (https://arxiv.org/pdf/1412.6071.pdf)
Usage: Copy code from layers as is. Pooling ration should be a tuple of 2 floats
between 1 and 2.
Adding a layer to model: model.add(FractionalMaxPooling2D((1.414, 1.414)))

from keras import Input, Model
from keras.constraints import max_norm
from keras.layers import GlobalAveragePooling2D, Reshape, Dense, multiply, GlobalMaxPooling2D, Add, Activation, Lambda, \
    Concatenate, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, AveragePooling2D, DepthwiseConv2D, \
    SeparableConv2D
from tensorflow.keras import backend as K
import numpy as np
import scipy.io as sio
from numpy import swapaxes
from keras import utils as np_utils
from keras.optimizers import Adam


def DeepConvNet(nb_classes, channels=3, Samples=4000,
                dropoutRate=0.5, kernLength=64, F1=8,
                D=2, F2=16, norm_rate=0.25):
    input_shape = (Samples, channels, 1)
    input_main = Input(input_shape)
    conv_filters = (2, 1)
    conv_filters2 = (1, channels)
    pool = (2, 1)
    strides = (2, 1)
    axis = -1

    block1 = Conv2D(25, conv_filters,
                    input_shape=input_shape,
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(input_main)
    block1 = Conv2D(25, conv_filters2,
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    block1 = BatchNormalization(axis=axis, epsilon=1e-05, momentum=0.1)(block1)
    block1 = Activation('elu')(block1)
    block1 = MaxPooling2D(pool_size=pool, strides=strides)(block1)
    block1 = Dropout(dropoutRate)(block1)

    block2 = Conv2D(50, conv_filters,
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    block2 = BatchNormalization(axis=axis, epsilon=1e-05, momentum=0.1)(block2)
    block2 = Activation('elu')(block2)
    block2 = MaxPooling2D(pool_size=pool, strides=strides)(block2)
    block2 = Dropout(dropoutRate)(block2)

    block3 = Conv2D(100, conv_filters,
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block2)
    block3 = BatchNormalization(axis=axis, epsilon=1e-05, momentum=0.1)(block3)
    block3 = Activation('elu')(block3)
    block3 = MaxPooling2D(pool_size=pool, strides=strides)(block3)
    block3 = Dropout(dropoutRate)(block3)

    block4 = Conv2D(200, conv_filters,
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block3)
    block4 = BatchNormalization(axis=axis, epsilon=1e-05, momentum=0.1)(block4)
    block4 = Activation('elu')(block4)
    block4 = MaxPooling2D(pool_size=pool, strides=strides)(block4)
    block4 = Dropout(dropoutRate)(block4)

    flatten = Flatten()(block4)

    dense = Dense(nb_classes, kernel_constraint=max_norm(0.5))(flatten)
    softmax = Activation('softmax')(dense)

    return Model(inputs=input_main, outputs=softmax)
def square(x):
    return K.square(x)

def log(x):
    return K.log(K.clip(x, min_value = 1e-7, max_value = 10000))
def ShallowConvNet(nb_classes, Chans=3, Samples=4000, dropoutRate=0.5):
    input_shape = (Samples, Chans, 1)
    conv_filters = (25, 1)
    conv_filters2 = (1, Chans)
    pool_size = (45, 1)
    strides = (15, 1)
    axis = -1
    input_main = Input(input_shape)
    block1 = Conv2D(20, conv_filters,
                    input_shape=input_shape,
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(input_main)
    block1 = Conv2D(20, conv_filters2, use_bias=False,
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    block1 = BatchNormalization(axis=axis, epsilon=1e-05, momentum=0.1)(block1)
    block1 = Activation('square')(block1)
    block1 = AveragePooling2D(pool_size=pool_size, strides=strides)(block1)
    block1 = Activation('log')(block1)
    block1 = Dropout(dropoutRate)(block1)
    flatten = Flatten()(block1)
    dense = Dense(nb_classes, kernel_constraint=max_norm(0.5))(flatten)
    softmax = Activation('softmax')(dense)
    model=Model(inputs=input_main, outputs=softmax)

    return model
def EEGNet(nb_classes, Chans=3, Samples=4000,
           dropoutRate=0.5, kernLength=64, F1=8,
           D=2, F2=16, norm_rate=0.25):

    input_shape = (Samples, Chans, 1)
    conv_filters = (kernLength, 1)
    depth_filters = (1, Chans)
    pool_size = (6, 1)
    pool_size2 = (12, 1)
    separable_filters = (20, 1)
    axis = -1
    input1 = Input(shape=input_shape)
    block1 = Conv2D(F1, conv_filters, padding='same',
                    input_shape=input_shape,
                    use_bias=False)(input1)
    block1 = BatchNormalization(axis=axis)(block1)
    block1 = DepthwiseConv2D(depth_filters, use_bias=False,
                             depth_multiplier=D,
                             depthwise_constraint=max_norm(1.))(block1)
    block1 = BatchNormalization(axis=axis)(block1)
    block1 = Activation('elu')(block1)
    block1 = AveragePooling2D(pool_size)(block1)
    block1 = Dropout(dropoutRate)(block1)

    block2 = SeparableConv2D(F2, separable_filters,
                             use_bias=False, padding='same')(block1)
    block2 = BatchNormalization(axis=axis)(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D(pool_size2)(block2)
    block2 = Dropout(dropoutRate)(block2)

    flatten = Flatten(name='flatten')(block2)

    dense = Dense(nb_classes, name='dense',
                  kernel_constraint=max_norm(norm_rate))(flatten)
    softmax = Activation('softmax', name='softmax')(dense)

    return Model(inputs=input1, outputs=softmax)
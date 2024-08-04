from keras import Input, Model
from keras.constraints import max_norm
from keras.layers import GlobalAveragePooling2D, Reshape, Dense, multiply, GlobalMaxPooling2D, Add, Activation, Lambda, \
    Concatenate, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, AveragePooling2D, DepthwiseConv2D, \
    SeparableConv2D
from tensorflow.keras import backend as K

def attach_attention_module(net, attention_module):#TODO 注意力模块
    if attention_module == 'se_block':  # SE_block
        net = se_block(net)
    elif attention_module == 'cbam_block':  # CBAM_block
        net = cbam_block(net)
    else:
        raise Exception("'{}' is not supported attention module!".format(attention_module))

    return net

def se_block(input_feature, ratio=2):#TODO SE模块
    channel_axis = -1
    channel = input_feature._keras_shape[channel_axis]

    se_feature = GlobalAveragePooling2D()(input_feature)
    se_feature = Reshape((1, 1, channel))(se_feature)
    se_feature = Dense(channel // ratio,
                       activation='relu',
                       kernel_initializer='he_normal',
                       use_bias=True,
                       bias_initializer='zeros')(se_feature)
    se_feature = Dense(channel,
                       activation='sigmoid',
                       kernel_initializer='he_normal',
                       use_bias=True,
                       bias_initializer='zeros')(se_feature)

    se_feature = multiply([input_feature, se_feature])
    return se_feature

def cbam_block(cbam_feature, ratio=2):#TODO CBAM模块
    cbam_feature = channel_attention(cbam_feature, ratio)
    cbam_feature = spatial_attention(cbam_feature)
    return cbam_feature

def channel_attention(input_feature, ratio=2):#TODO 使用注意力模块计算各通道权重
    channel_axis = -1
    channel = K.int_shape(input_feature)[channel_axis]

    shared_layer_one = Dense(channel // ratio,
                             activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)

    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1, 1, channel))(max_pool)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)

    cbam_feature = Add()([avg_pool, max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)
    return multiply([input_feature, cbam_feature])

def spatial_attention(input_feature):#TODO 使用空间注意力模块计算各位置权重
    kernel_size = 7
    cbam_feature = input_feature

    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    cbam_feature = Conv2D(filters=1,
                          kernel_size=kernel_size,
                          strides=1,
                          padding='same',
                          activation='sigmoid',
                          kernel_initializer='he_normal',
                          use_bias=False)(concat)

    return multiply([input_feature, cbam_feature])

def CNN_CBAM(nb_classes, Chans=3, Samples=4000):#TODO 构建带有CBAM的CNN
    input_shape = (Samples, 1, Chans)
    conv_filters = (100, 1)
    pool_size = (3, 1)
    strides = (3, 1)
    input_main = Input(input_shape)

    block1 = Conv2D(25, conv_filters, input_shape=input_shape, padding='same', activation='relu')(input_main)
    block1 = attach_attention_module(block1, attention_module='cbam_block')
    block1 = MaxPooling2D(pool_size=pool_size, strides=strides)(block1)
    block1 = BatchNormalization()(block1)
    block1 = Dropout(0.5)(block1)

    block2 = Conv2D(50, conv_filters, padding='same', activation='relu')(block1)
    block2 = MaxPooling2D(pool_size=pool_size, strides=strides)(block2)
    block2 = BatchNormalization()(block2)
    block2 = Dropout(0.5)(block2)

    block3 = Conv2D(75, conv_filters, padding='same', activation='relu')(block2)
    block3 = MaxPooling2D(pool_size=pool_size, strides=strides)(block3)
    block3 = BatchNormalization()(block3)
    block3 = Dropout(0.5)(block3)

    block4 = Conv2D(100, conv_filters, padding='same', activation='relu')(block3)
    block4 = MaxPooling2D(pool_size=pool_size, strides=strides)(block4)
    block4 = BatchNormalization()(block4)
    block4 = Dropout(0.5)(block4)
    flatten = Flatten()(block4)
    dense = Dense(nb_classes)(flatten)
    softmax = Activation('softmax')(dense)
    model = Model(inputs=input_main, outputs=softmax)

    return model

def CNN(nb_classes, Chans=3, Samples=4000):#TODO 构建不带有CBAM的CNN
    input_shape = (Samples, 1, Chans)
    conv_filters = (100, 1)
    pool_size = (3, 1)
    strides = (3, 1)
    input_main = Input(input_shape)

    block1 = Conv2D(25, conv_filters, input_shape=input_shape, padding='same', activation='relu')(input_main)
    block1 = MaxPooling2D(pool_size=pool_size, strides=strides)(block1)
    block1 = BatchNormalization()(block1)
    block1 = Dropout(0.5)(block1)

    block2 = Conv2D(50, conv_filters, padding='same', activation='relu')(block1)
    block2 = MaxPooling2D(pool_size=pool_size, strides=strides)(block2)
    block2 = BatchNormalization()(block2)
    block2 = Dropout(0.5)(block2)

    block3 = Conv2D(75, conv_filters, padding='same', activation='relu')(block2)
    block3 = MaxPooling2D(pool_size=pool_size, strides=strides)(block3)
    block3 = BatchNormalization()(block3)
    block3 = Dropout(0.5)(block3)

    block4 = Conv2D(100, conv_filters, padding='same', activation='relu')(block3)
    block4 = MaxPooling2D(pool_size=pool_size, strides=strides)(block4)
    block4 = BatchNormalization()(block4)
    block4 = Dropout(0.5)(block4)
    flatten = Flatten()(block4)

    dense = Dense(nb_classes)(flatten)
    softmax = Activation('softmax')(dense)
    model = Model(inputs=input_main, outputs=softmax)

    return model

def EEGNet(nb_classes, Chans=3, Samples=4000,#TODO 构建EEGNet
           dropoutRate=0.5, kernLength=1000, F1=8,
           D=2, F2=16, norm_rate=0.25, cpu=False):

    input_shape = (Samples, Chans, 1)
    conv_filters = (kernLength, 1)
    depth_filters = (1, Chans)
    pool_size = (9, 1)
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

def Block1(x):#TODO 定义卷积模块1
    block1 = Conv2D(25, (25,1), input_shape=(4000, 1 ,3), padding='same', activation='relu')(x)
    block1 = attach_attention_module(block1, attention_module='se_block')
    block1 = MaxPooling2D(pool_size=(3,1), strides=(3,1))(block1)
    block1 = BatchNormalization()(block1)
    block1 = Dropout(0.5)(block1)

    block2 = Conv2D(50, (25,1), padding='same', activation='relu')(block1)
    block2 = MaxPooling2D(pool_size=(3,1), strides=(3,1))(block2)
    block2 = BatchNormalization()(block2)
    block2 = Dropout(0.5)(block2)

    block3 = Conv2D(75, (25,1), padding='same', activation='relu')(block2)
    block3 = MaxPooling2D(pool_size=(3,1), strides=(3,1))(block3)
    block3 = BatchNormalization()(block3)
    block3 = Dropout(0.5)(block3)

    block4 = Conv2D(100, (25,1), padding='same', activation='relu')(block3)
    block4 = MaxPooling2D(pool_size=(3,1), strides=(3,1))(block4)
    block4 = BatchNormalization()(block4)
    block4 = Dropout(0.5)(block4)
    x = Flatten()(block4)
    return x


def Block2(x):#TODO 定义卷积模块2
    block1 = Conv2D(25, (50,1), input_shape=(4000, 1 ,3), padding='same', activation='relu')(x)
    block1 = attach_attention_module(block1, attention_module='se_block')
    block1 = MaxPooling2D(pool_size=(3,1), strides=(3,1))(block1)
    block1 = BatchNormalization()(block1)
    block1 = Dropout(0.5)(block1)

    block2 = Conv2D(50,(50,1), padding='same', activation='relu')(block1)
    #     block2=attach_attention_module(block2,attention_module='se_block')
    block2 = MaxPooling2D(pool_size=(3,1), strides=(3,1))(block2)
    block2 = BatchNormalization()(block2)
    block2 = Dropout(0.5)(block2)

    block3 = Conv2D(75, (50,1), padding='same', activation='relu')(block2)
    block3 = MaxPooling2D(pool_size=(3,1), strides=(3,1))(block3)
    block3 = BatchNormalization()(block3)
    block3 = Dropout(0.5)(block3)

    block4 = Conv2D(100, (50,1), padding='same', activation='relu')(block3)
    block4 = MaxPooling2D(pool_size=(3,1), strides=(3,1))(block4)
    block4 = BatchNormalization()(block4)
    block4 = Dropout(0.5)(block4)
    x = Flatten()(block4)
    return x


def Block3(x):#TODO 定义卷积模块3
    block1 = Conv2D(25, (100,1), input_shape=(4000, 1 ,3), padding='same', activation='relu')(x)
    block1 = attach_attention_module(block1, attention_module='se_block')
    block1 = MaxPooling2D(pool_size=(3,1), strides=(3,1))(block1)
    block1 = BatchNormalization()(block1)
    block1 = Dropout(0.5)(block1)

    block2 = Conv2D(50, (100,1), padding='same', activation='relu')(block1)
    block2 = MaxPooling2D(pool_size=(3,1), strides=(3,1))(block2)
    block2 = BatchNormalization()(block2)
    block2 = Dropout(0.5)(block2)

    block3 = Conv2D(75, (100,1), padding='same', activation='relu')(block2)
    #     block3=attach_attention_module(block3,attention_module='se_block')
    block3 = MaxPooling2D(pool_size=(3,1), strides=(3,1))(block3)
    block3 = BatchNormalization()(block3)
    block3 = Dropout(0.5)(block3)

    block4 = Conv2D(100, (100,1), padding='same', activation='relu')(block3)
    block4 = MaxPooling2D(pool_size=(3,1), strides=(3,1))(block4)
    block4 = BatchNormalization()(block4)
    block4 = Dropout(0.5)(block4)
    x = Flatten()(block4)
    return x


def Block4(x):#TODO 定义卷积模块4
    block1 = Conv2D(25, (200,1), input_shape=(4000, 1 ,3), padding='same', activation='relu')(x)
    block1 = attach_attention_module(block1, attention_module='se_block')
    block1 = MaxPooling2D(pool_size=(3,1), strides=(3,1))(block1)
    block1 = BatchNormalization()(block1)
    block1 = Dropout(0.5)(block1)

    block2 = Conv2D(50, (200,1), padding='same', activation='relu')(block1)
    block2 = MaxPooling2D(pool_size=(3,1), strides=(3,1))(block2)
    block2 = BatchNormalization()(block2)
    block2 = Dropout(0.5)(block2)

    block3 = Conv2D(75, (200,1), padding='same', activation='relu')(block2)
    block3 = MaxPooling2D(pool_size=(3,1), strides=(3,1))(block3)
    block3 = BatchNormalization()(block3)
    block3 = Dropout(0.5)(block3)

    block4 = Conv2D(100, (200,1), padding='same', activation='relu')(block3)
    #     block4=attach_attention_module(block4,attention_module='se_block')
    block4 = MaxPooling2D(pool_size=(3,1), strides=(3,1))(block4)
    block4 = BatchNormalization()(block4)
    block4 = Dropout(0.5)(block4)
    x = Flatten()(block4)
    return x


def FAModel(nb_classes):#TODO 将所有卷积结果输出  连接，利用全连接和softmax激活函数  分类
    input_main = Input((4000,1 ,3))
    block1 = Block1(input_main)
    block2 = Block2(input_main)
    block3 = Block3(input_main)
    block4 = Block4(input_main)
    x = Concatenate(axis=-1)([block1, block2, block3, block4])
    x = Dense(nb_classes)(x)
    softmax = Activation('softmax')(x)
    model = Model(inputs=input_main, outputs=softmax)
    return model


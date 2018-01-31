# -*- coding: utf-8 -*-
# @Time    : 2018/1/31 10:56
# @Author  : HLin
# @Email   : linhua2017@ia.ac.cn
# @File    : VGG19.py
# @Software: PyCharm

from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential
import keras.backend as K

K.set_image_data_format('channels_last')
K.set_learning_phase(1)

def VGG_19(weight_path = None):
    model = Sequential()
    # block 1
    model.add(Conv2D(filters=64,
                     kernel_size=(3,3),
                     strides=(1,1),
                     padding='same',
                     activation='relu',
                     input_shape=(224, 224, 3),
                     name='block1_conv1'))
    model.add(Conv2D(filters=64,
                     kernel_size=(3,3),
                     strides=(1,1),
                     padding='same',
                     activation='relu',
                     name='block1_conv2'))
    model.add(MaxPooling2D(pool_size=(2,2),
                           strides=(2,2),
                           name='block1_pool'))


    # block 2
    model.add(Conv2D(filters=128,
                     kernel_size=(3, 3),
                     strides=(1, 1),
                     padding='same',
                     activation='relu',
                     name='block2_conv1'))
    model.add(Conv2D(filters=128,
                     kernel_size=(3, 3),
                     strides=(1, 1),
                     padding='same',
                     activation='relu',
                     name='block2_conv2'))
    model.add(MaxPooling2D(pool_size=(2, 2),
                           strides=(2,2),
                           name='block2_pool'))


    # block 3
    model.add(Conv2D(filters=256,
                     kernel_size=(3, 3),
                     strides=(1, 1),
                     padding='same',
                     activation='relu',
                     name='block3_conv1'))
    model.add(Conv2D(filters=256,
                     kernel_size=(3, 3),
                     strides=(1, 1),
                     padding='same',
                     activation='relu',
                     name='block3_conv2'))
    model.add(Conv2D(filters=256,
                     kernel_size=(3, 3),
                     strides=(1, 1),
                     padding='same',
                     activation='relu',
                     name='block3_conv3'))
    model.add(Conv2D(filters=256,
                     kernel_size=(3, 3),
                     strides=(1, 1),
                     padding='same',
                     activation='relu',
                     name='block3_conv4'))
    model.add(MaxPooling2D(pool_size=(2, 2),
                           strides=(2,2),
                           name='block3_pool'))


    # bvlock 4
    model.add(Conv2D(filters=512,
                     kernel_size=(3, 3),
                     strides=(1, 1),
                     padding='same',
                     activation='relu',
                     name='block4_conv1'))
    model.add(Conv2D(filters=512,
                     kernel_size=(3, 3),
                     strides=(1, 1),
                     padding='same',
                     activation='relu',
                     name='block4_conv2'))
    model.add(Conv2D(filters=512,
                     kernel_size=(3, 3),
                     strides=(1, 1),
                     padding='same',
                     activation='relu',
                     name='block4_conv3'))
    model.add(Conv2D(filters=512,
                     kernel_size=(3, 3),
                     strides=(1, 1),
                     padding='same',
                     activation='relu',
                     name='block4_conv4'))
    model.add(MaxPooling2D(pool_size=(2, 2),
                           strides=(2,2),
                           name='block4_pool'))


    # block 5
    model.add(Conv2D(filters=512,
                     kernel_size=(3, 3),
                     strides=(1, 1),
                     padding='same',
                     activation='relu',
                     name='block5_conv1'))
    model.add(Conv2D(filters=512,
                     kernel_size=(3, 3),
                     strides=(1, 1),
                     padding='same',
                     activation='relu',
                     name='block5_conv2'))
    model.add(Conv2D(filters=512,
                     kernel_size=(3, 3),
                     strides=(1, 1),
                     padding='same',
                     activation='relu',
                     name='block5_conv3'))
    model.add(Conv2D(filters=512,
                     kernel_size=(3, 3),
                     strides=(1, 1),
                     padding='same',
                     activation='relu',
                     name='block5_conv4'))
    model.add(MaxPooling2D(pool_size=(2, 2),
                           strides=(2,2),
                           name='block5_pool'))
    model.add(Flatten())
    model.add(Dense(units=4096,activation='relu', name='fc1'))
    #model.add(Dropout(0.5))
    model.add(Dense(units=4096,activation='relu', name='fc2'))
    #model.add(Dropout(0.5))
    model.add(Dense(units=1000,activation='softmax', name='predictions'))

    if weight_path :
        model.load_weights(weight_path)
    return model




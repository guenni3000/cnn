import numpy as np
from keras import backend as K
import matplotlib.pyplot as plt
import skimage
from skimage import io, transform
import tensorflow as tf
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Input, Lambda
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import os
import keras

dir = "C:/Users/Julian/Desktop/set/output"
in_shape = (187, 250, 1)
comp_filter_size = 8
compare_filters = 5
batch_size = 1

def compare(x):
    dims = x.get_shape()

    imgs1 = x[:, :, :, 0:int(int(dims[3])/2)]
    imgs2 = x[:, :, :, int(int(dims[3])/2):]

    imgs_out = []

    for b in range(batch_size):
        img1 = imgs1[b]
        img2 = tf.expand_dims(imgs2[b], 0)

        filters_k = []

        for f in range(compare_filters):
            
            filters = []

            for i in range(int(int(dims[3])/2)):
                x_coord = np.random.randint(0, int(dims[1])-comp_filter_size)
                y_coord = np.random.randint(0, int(dims[2])-comp_filter_size)
                filter_new = img1[x_coord:x_coord+comp_filter_size, y_coord:y_coord+comp_filter_size, i]

                filters.append(filter_new)
            
            filters_k.append(tf.stack(filters, axis=2))

        img_new = tf.nn.conv2d(img2, tf.stack(filters_k, 3), [1, 1, 1, 1], 'SAME')
        imgs_out.append(img_new)

    return tf.concat(imgs_out, 0)

def loadimg():
    img1 = io.imread(dir+'/0000_0_0.png', as_gray=True)
    img1 = np.expand_dims(img1, -1)

    img2 = io.imread(dir+'/0000_1_0.png', as_gray=True)
    img2 = np.expand_dims(img2, -1)

    img_base = io.imread(dir+'/0000_2_0.png', as_gray=True)
    img_base = np.expand_dims(img_base, -1)
    return img1[np.newaxis, :], img2[np.newaxis, :], img_base[np.newaxis, :]

input1 = Input(shape=in_shape, name='input1')
input2 = Input(shape=in_shape, name='input2')
input_base = Input(shape=in_shape, name='input_base')

conv_layer = Conv2D(16, (3, 3), padding='same', activation='relu')

conv1 = conv_layer(input1)
conv2 = conv_layer(input2)
conv_base = conv_layer(input_base)

pooling_layer = MaxPooling2D(pool_size=(4, 4))

pool1 = pooling_layer(conv1)
pool2 = pooling_layer(conv2)
pool_base = pooling_layer(conv_base)

merged1 = keras.layers.concatenate([pool1, pool_base], axis=-1)
merged2 = keras.layers.concatenate([pool2, pool_base], axis=-1)

merged1 = Dropout(0.25)(merged1)
merged2 = Dropout(0.25)(merged2)

compare_layer = Lambda(compare)

comp1 = compare_layer(merged1)
comp2 = compare_layer(merged2)

comp1 = pooling_layer(comp1)
comp2 = pooling_layer(comp2)

comp1 = Dropout(0.25)(comp1)
comp2 = Dropout(0.25)(comp2)

comp1 = Flatten()(comp1)
comp2 = Flatten()(comp2)

merged_final = keras.layers.concatenate([comp1, comp2], axis=-1)

x = Dense(64, activation='relu')(merged_final)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)

output = Dense(1, activation='sigmoid', name='output')(x)

model = Model(inputs=[input1, input2, input_base], outputs=[output])
model.compile(optimizer='rmsprop', loss='binary_crossentropy')

#for layer in model.layers:
#    print(layer.get_output_at(0))
                              
functor = K.function([model.layers[0].input, model.layers[1].input, model.layers[6].input, K.learning_phase()], [model.layers[7].get_output_at(2)])

test1, test2, test_base = loadimg()
outs = functor([test1, test2, test_base, 0])

outs = outs[0][0]

outs = np.swapaxes(outs, 1, 2)
outs = np.swapaxes(outs, 0, 1)

for i in range(16):
    print(np.shape(outs[i]))

    #print(np.shape(outs))

    plt.imshow(outs[i])
    plt.show()


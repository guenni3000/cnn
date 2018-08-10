import numpy as np
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
batch_size = 8
file_count = len(os.listdir(os.fsencode(dir)))

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

def get_imgs(path):
    while True:
        in1 = []
        in2 = []
        in_base = []
        out = []

        for file in os.listdir(os.fsencode(path)):
            filename = os.fsdecode(file)
            parts = filename.split('_')

            if parts[1] == '0':
                output = int(parts[2][0])
                path1 = path+'/'+filename
                path2 = path+'/'+parts[0]+'_1_'+parts[2]
                path3 = path+'/'+parts[0]+'_2_'+parts[2]

                img1 = io.imread(path1, as_gray=True)
                #img1 = skimage.img_as_ubyte(img1)
                img1 = np.expand_dims(img1, -1)

                img2 = io.imread(path2, as_gray=True)
                #img2 = skimage.img_as_ubyte(img2)
                img2 = np.expand_dims(img2, -1)

                img3 = io.imread(path3, as_gray=True)
                #img3 = skimage.img_as_ubyte(img3)
                img3 = np.expand_dims(img3, -1)

                in1.append(img1)
                in2.append(img2)
                in_base.append(img3)
                out.append(output)

            if len(in1) == batch_size:
                yield ([np.stack(in1), np.stack(in2), np.stack(in_base)], np.stack(out))
                in1 = []
                in2 = []
                in_base = []
                out = []

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
model.fit_generator(get_imgs(dir), steps_per_epoch=np.array([int(file_count/batch_size)]), epochs=10)

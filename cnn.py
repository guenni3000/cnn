import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Input, Lambda
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import os
import keras

dir = "C:/Users/Julian/Desktop/dataset/training"
in_shape = (500, 375, 1)
comp_filter_size = 5
compare_filters = 5
file_count = len(os.listdir(os.fsencode(dir)))

def compare(x):
    dims = tf.shape(x)

    imgs1 = x[:][:][:][0:dims[3]/2]
    imgs2 = x[:][:][:][dims[3]/2:]

    imgs_out = []

    for b in range(dims[0]):
        img1 = imgs1[b]
        img2 = tf.expand_dims(imgs2[b], 0)

        filters_k = []

        for f in range(comp_filter_size):
            
            for i in range(dims[3]/2):
                x_coord = np.random.randint(0, dims[1]-comp_filter_size)
                y_coord = np.random.randint(0, dims[2]-comp_filter_size)
                filter_new = img1[x_coord:x_coord+comp_filter_size][y_coord:y_coord+comp_filter_size][i]

                if i == 0:
                    filters = filter_new
                else:
                    filters = np.concatenate((filters, filter_new), 2)
            
            filters_k.append(filters)

        img_new = tf.nn.conv2d(img2, tf.stack(filters_k, 3), [1, 1, 1, 1], 'SAME')
        imgs_out.append(img_new)

    return tf.parallel_stack(imgs_out)

def get_imgs(path):
    for file in os.listdir(os.fsencode(path)):
        filename = os.fsdecode(file)
        parts = filename.split('_')
        
        if parts[1] == '0':
            out = parts[2][0]
            path1 = path+'/'+filename
            path2 = path+'/'+parts[0]+'_1_'+parts[2]
            path3 = path+'/'+parts[0]+'_2_'+parts[2]

            yield ({'input1': tf.image.decode_image(tf.read_file(path1), 1), 
                    'input2': tf.image.decode_image(tf.read_file(path2), 1), 
                    'input_base': tf.image.decode_image(tf.read_file(path3), 1)}, {'output': out})

input1 = Input(shape=in_shape, dtype='uint8', name='input1')
input2 = Input(shape=in_shape, dtype='uint8', name='input2')
input_base = Input(shape=in_shape, dtype='uint8', name='input_base')

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

comp1 = Flatten()(comp1)
comp2 = Flatten()(comp2)

merged_final = keras.layers.concatenate([comp1, comp2], axis=-1)

x = Dense(64, activation='relu')(merged_final)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)

output = Dense(1, activation='sigmoid', name='output')(x)

model = Model(inputs=[input1, input2], outputs=[output])
model.compile(optimizer='rmsprop', loss='binary_crossentropy')
model.fit_generator(get_imgs(dir), steps_per_epoch=file_count/8, epochs=10)

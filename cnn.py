import numpy as np
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Input, Lambda
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import os
import keras

dir = "C:/Users/Julian/Desktop/dataset/training"
in_shape = (500, 375, 1)
file_count = len(os.listdir(os.fsencode(dir)))

def compare(x):
    img1 = x[0:len(x)][0:len(x[0])][0]
    img2 = x[0:len(x)][0:len(x[0])][1]



def get_imgs(path):
    for file in os.listdir(os.fsencode(path)):
        filename = os.fsdecode(file)
        parts = filename.split('_')
        if parts[1] == '0':
            out = parts[2][0]
            img1 = Image.open(path+'/'+filename) 
            img2 = Image.open(path+'/'+parts[0]+'_1_'+parts[2])
            img3 = Image.open(path+'/'+parts[0]+'_2_'+parts[2])
            yield ({'input1': img1.load(), 'input2': img2.load(), 'input_base': img3.load()}, {'output': out})

input1 = Input(shape=in_shape, dtype='int32', name='input1')
input2 = Input(shape=in_shape, dtype='int32', name='input2')
input_base = Input(shape=in_shape, dtype='int32', name='input_base')

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

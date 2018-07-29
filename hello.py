from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import os

dir = "C:/Users/Julian/Desktop/dataset"

def get_imgs(path):
    for file in os.listdir(os.fsencode(path)):
        filename = os.fsdecode(file)
        parts = filename.split('_')
        if parts[1] == '0':
            out = parts[2][0]
            img1 = Image.open(path+'/'+filename) 
            img2 = Image.open(path+'/'+parts[0]+'_1_'+parts[2])
            img3 = Image.open(path+'/'+parts[0]+'_2_'+parts[2])
            yield ({'input_1': img1.load(), 'input_2': img2.load(), 'input_3': img3.load()}, {'output': out})

train_iterator_base = get_imgs(dir+'/trainingBase')

for img in train_iterator_base:
    print(img)
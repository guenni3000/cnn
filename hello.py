from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import os

dir = "C:/Users/Julian/Desktop/dataset"
max_size = 500

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

img = Image.open(dir+'/training/'+os.fsdecode(os.listdir(os.fsencode(dir+'/training/'))[0]))

height = img.height
width = img.width

ratio = (width*1.0)/height

if height > width:
    height = max_size
    width = int(max_size*ratio)
else:
    width = max_size
    height = int(max_size/ratio)

ratio = (img.width*1.0)/img.height

datagen = ImageDataGenerator()

train_iterator_base = datagen.flow_from_directory(dir, target_size=(height, width), color_mode='grayscale', class_mode='binary', shuffle=False)

print(train_iterator_base.image_shape)
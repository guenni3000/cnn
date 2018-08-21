import numpy as np
import skimage
import skimage.io as io
import matplotlib.pyplot as plt

filter1 = []

def max_pooling(arr, strides):
    dims = np.shape(arr)
    output = np.empty([int(dims[0]/strides), int(dims[1]/strides)])

    dims_new = np.shape(output)

    for x in range(dims_new[0]):
        for y in range(dims_new[1]):
            max = 0
            for x1 in range(strides):
                for y1 in range(strides):
                    if arr[x*strides+x1][y*strides+y1] > max:
                        max = arr[x*strides+x1][y*strides+y1]
            output[x][y] = max

    return output

def conv2d(arr, filter_size):
    dims = np.shape(arr)
    pad = np.zeros([dims[0]+filter_size-dims[0]%filter_size+2, dims[1]+filter_size-dims[1]%filter_size+2])
    pad[:dims[0],:dims[1]] = arr

    if filter1 == []:
        filter = np.random.randint(-2, 3, size=(filter_size, filter_size))
    else:
        filter = filter1

    out = np.empty(dims, np.float)

    for x in range(dims[0]):
        for y in range(dims[1]):
            sum = 0
            for x1 in range(filter_size):
                for y1 in range(filter_size):
                    sum += pad[x+x1][y+y1]*filter[x1][y1]
            out[x][y] = sum

    return out, filter

def compare(arr, comp, filter_size, tolerance):
    dims = np.shape(arr)

    dims = np.shape(arr)
    pad = np.zeros([dims[0]+filter_size-dims[0]%filter_size+5, dims[1]+filter_size-dims[1]%filter_size+5])
    pad[:dims[0],:dims[1]] = arr

    x_coord = np.random.randint(0, dims[0]-filter_size)
    y_coord = np.random.randint(0, dims[1]-filter_size)
    filter = comp[x_coord:x_coord+filter_size, y_coord:y_coord+filter_size]

    out = np.empty(dims)

    for x in range(dims[0]):
        for y in range(dims[1]):
            sum = 0
            for x1 in range(filter_size):
                for y1 in range(filter_size):
                    if (pad[x+x1][y+y1]-filter[x1][y1])**2 < tolerance**2:
                        sum += 1/filter_size**2
            out[x][y] = sum

    return out

def normalize(arr):
    arr_min = np.min(arr)
    arr_max = np.max(arr)

    arr = np.subtract(arr, arr_min)
    arr = np.divide(arr, arr_max - arr_min)

    return arr

img1 = io.imread('C:/Users/Julian/Desktop/set/output/0000_0_0.png', as_gray=True)
img2 = io.imread('C:/Users/Julian/Desktop/set/output/0000_1_0.png', as_gray=True)
img3 = io.imread('C:/Users/Julian/Desktop/set/output/0000_2_0.png', as_gray=True)

#img1 = max_pooling(img1, 3)
#img2 = max_pooling(img2, 3)
#img3 = max_pooling(img3, 3)

#img1, filter = conv2d(img1, 4)
#filter1 = filter
#img2, filter = conv2d(img2, 4)
#img3, filter = conv2d(img3, 4)

#print(img1)
#print(normalize(img1))

img1 = max_pooling(normalize(img1), 2)
img2 = max_pooling(normalize(img2), 2)
img3 = max_pooling(normalize(img3), 2)

plt.imshow(img1)
plt.show()

out1 = compare(img2, img1, 5, 0.05)
out2 = compare(img2, img3, 5, 0.05)

plt.imshow(normalize(out1))
plt.show()

plt.imshow(normalize(out2))
plt.show()
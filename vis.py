#from _future_ import print_function
from keras.constraints import maxnorm
from keras.optimizers import SGD


import cv2
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from scipy.misc import imsave
import numpy as np
import time
#from keras.applications import vgg16
from keras import backend as K
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D

num_classes=2

img_width=36
img_height=36
ch=3
filter_nos=32
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    # if K.image_data_format() == 'channels_first':
    #     x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x
def baseline_model():
    # create model
    model = Sequential()
    model.add(Conv2D(32,3, 3, border_mode='same', input_shape=(36,36, 3), activation='relu', name='layer1'))
    #model.add(Dropout(0.15))
    model.add(Conv2D(32,3, 3, activation='relu', border_mode='same', name= 'layer2'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64,3, 3, activation='relu', border_mode='same', name= 'layer3'))
    #model.add(Dropout(0.15))
    model.add(Conv2D(64,3, 3, activation='relu', border_mode='same',  name= 'layer4'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128,3, 3, activation='relu', border_mode='same', name= 'layer5'))
    #model.add(Dropout(0.15))
    model.add(Conv2D(128,3, 3, activation='relu', border_mode='same', name= 'layer6'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    #model.add(Dropout(0.15))
    model.add(Dense(1024, activation='relu', W_constraint=maxnorm(3)))
    #model.add(Dropout(0.15))
    model.add(Dense(512, activation='relu', W_constraint=maxnorm(3)))
    #model.add(Dropout(0.15))
    model.add(Dense(num_classes))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    return model


model = baseline_model()
model.load_weights("model.h5")
model.summary()



layer_name="layer1"

# this is the placeholder for the input images
input_img = model.input


# get the symbolic outputs of each "key" layer (we gave them unique names).
layer_dict = dict([(layer.name, layer) for layer in model.layers[:]])
print(layer_dict.keys())
def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


kept_filters = []
for filter_index in range(filter_nos):
    # we only scan through the first 200 filters,
    # but there are actually 512 of them
    print('Processing filter %d' % filter_index)
    start_time = time.time()

    # we build a loss function that maximizes the activation
    # of the nth filter of the layer considered
    layer_output = layer_dict[layer_name].output
    if K.image_data_format() == 'channels_first':
        loss = K.mean(layer_output[:, filter_index, :, :])
    else:
        loss = K.mean(layer_output[:, :, :, filter_index])


    # we compute the gradient of the input picture wrt this loss
    grads = K.gradients(loss, input_img)[0]

    # normalization trick: we normalize the gradient
    grads = normalize(grads)

    # this function returns the loss and grads given the input picture
    iterate = K.function([input_img], [loss, grads])

    # step size for gradient ascent
    step = 1.0001

     # we start from a gray image with some random noise
    # if K.image_data_format() == 'channels_first':
    #     input_img_data = np.random.random((1, 1, img_width, img_height))
    # else:
    #     input_img_data = np.random.random((1, img_width, img_height, 1))
    # input_img_data = (input_img_data - 0.5) * 20 + 128
    #Giving an input:
    input_img_data=np.zeros((1,36,36,3))
    im=cv2.imread('35000.jpg')
    im=cv2.resize(im,(36,36))
    input_img_data[0,:,:,:]=im


    # we run gradient ascent for 20 steps
    for i in range(20):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step

        # print('Current loss value:', loss_value)
        if loss_value <= 0.:
            # some filters get stuck to 0, we can skip them
            break
    # print('Final loss : ',loss_value)


    # decode the resulting input image
    if loss_value > 0:
        img = deprocess_image(input_img_data[0])
        kept_filters.append((img, loss_value))
    end_time = time.time()
    print('Filter %d processed in %ds' % (filter_index, end_time - start_time))


n=4
# the filters that have the highest loss are assumed to be better-looking.
# we will only keep the top 64 filters.
margin = 5
width = n * img_width + (n - 1) * margin
height = n * img_height + (n - 1) * margin
stitched_filters = np.zeros((width, height, 3))

# fill the picture with our saved filters
for i in range(n):
    for j in range(n):
        img, loss = kept_filters[i * n + j]
        stitched_filters[(img_width + margin) * i: (img_width + margin) * i + img_width,
                         (img_height + margin) * j: (img_height + margin) * j + img_height, :] = img

# save the result to disk

imsave('L1.png', stitched_filters)

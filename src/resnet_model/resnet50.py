from __future__ import division, print_function, absolute_import

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tflearn
from tflearn.data_utils import image_preloader
import numpy as np
import rasterio
import cv2
from scipy.misc import imread, imresize
import csv
from PIL import Image

def to_categorical(y, nb_classes):
    y = np.asarray(y, dtype='int32')
    # high dimensional array warning
    if len(y.shape) > 2:
        warnings.warn('{}-dimensional array is used as input array.'.format(len(y.shape)), stacklevel=2)
    # flatten high dimensional array
    if len(y.shape) > 1:
        y = y.reshape(-1)
    if not nb_classes:
        nb_classes = np.max(y)+1
    Y = np.zeros((len(y), nb_classes))
    Y[np.arange(len(y)),y] = 1.
    return Y
    
def load_image(in_image):
    """ Load an image, returns PIL.Image. """
    # if the path appears to be an URL
    
    img = Image.open(in_image)
    return img


def resize_image(in_image, new_width, new_height, out_image=None,
                 resize_mode=Image.ANTIALIAS):

    img = in_image.resize((new_width, new_height), resize_mode)
    if out_image:
        img.save(out_image)
    return img


def convert_color(in_image, mode):
    """ Convert image color with provided `mode`. """
    return in_image.convert(mode)


def pil_to_nparray(pil_image):
    """ Convert a PIL.Image to numpy array. """
    pil_image.load()
    return np.asarray(pil_image, dtype="float32")

class Preloader(object):
    def __init__(self, array, function):
        self.array = array
        self.function = function

    def __getitem__(self, id):
        if type(id) in [list, np.ndarray]:
            return [self.function(self.array[i]) for i in id]
        elif isinstance(id, slice):
            return [self.function(arr) for arr in self.array[id]]
        else:
            return self.function(self.array[id])

    def __len__(self):
        return len(self.array)

class ImagePreloader(Preloader):
    def __init__(self, array, image_shape, normalize=True, grayscale=False):
        fn = lambda x: self.preload(x, image_shape, normalize, grayscale)
        super(ImagePreloader, self).__init__(array, fn)

    def preload(self, path, image_shape, normalize=True, grayscale=False):
        img = load_image(path)
        width, height = img.size
        if width != image_shape[0] or height != image_shape[1]:
            img = resize_image(img, image_shape[0], image_shape[1])
        if grayscale:
            img = convert_color(img, 'L')
        img = pil_to_nparray(img)
        if grayscale:
            img = np.reshape(img, img.shape + (1,))
        if normalize:
            img /= 255.
        return img


class LabelPreloader(Preloader):
    def __init__(self, array, n_class=None, categorical_label=True):
        fn = lambda x: self.preload(x, n_class, categorical_label)
        super(LabelPreloader, self).__init__(array, fn)

    def preload(self, label, n_class, categorical_label):
        if categorical_label:
            #TODO: inspect assert bug
            #assert isinstance(n_class, int)
            return to_categorical([label], n_class)[0]
        else:
            return label

def image_preloader(target_path, image_shape, mode='file', normalize=True,
                    grayscale=False, categorical_labels=True,
                    files_extension=None, filter_channel=False):

    assert mode in ['folder', 'file']
    if mode == 'folder':
        images, labels = directory_to_samples(target_path,
                                              flags=files_extension, filter_channel=filter_channel)
    else:
        with open(target_path, 'r') as f:
            images, labels = [], []
            for l in f.readlines():
                l = l.strip('\n').split()
                if not files_extension or any(flag in l[0] for flag in files_extension):
                    if filter_channel:
                        if get_img_channel(l[0]) != 3:
                            continue
                    images.append(l[0])
                    labels.append(int(l[1]))

    n_classes = np.max(labels) + 1
    X = ImagePreloader(images, image_shape, normalize, grayscale)
    Y = LabelPreloader(labels, n_classes, categorical_labels)

    return X, Y

tflearn.config.init_graph (num_cores=4, gpu_memory_fraction=0.9)

X_train, y_train = image_preloader('planet_cloudless_train.csv',image_shape=(64,64), mode='file', categorical_labels=True, normalize=False)
X_test, y_test = image_preloader('planet_cloudless_test.csv',image_shape=(64,64), mode='file', categorical_labels=True, normalize=False)

print('Data loaded')

n = 5

# Real-time data preprocessing
img_prep = tflearn.ImagePreprocessing()
img_prep.add_featurewise_zero_center([96.95961247, 126.28265043, 153.15617433], per_channel=True)


# Real-time data augmentation
img_aug = tflearn.ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_90degrees_rotation(rotations=[0,1,2,3])
img_aug.add_random_flip_updown()

# Building Residual Network
net = tflearn.input_data(shape=[None, 64, 64, 3], data_preprocessing=img_prep, data_augmentation=img_aug)
net = tflearn.conv_2d(net, 16, 3, regularizer='L2', weight_decay=0.0001)
net = tflearn.residual_block(net, n, 16)
net = tflearn.residual_block(net, 1, 32, downsample=True)
net = tflearn.residual_block(net, n-1, 32)
net = tflearn.residual_block(net, 1, 64, downsample=True)
net = tflearn.residual_block(net, n-1, 64)
net = tflearn.batch_normalization(net)
net = tflearn.activation(net, 'relu')
net = tflearn.global_avg_pool(net)

# Regression
net = tflearn.fully_connected(net, 2, activation='softmax')
mom = tflearn.Momentum(0.1, lr_decay=0.1, decay_step=32000, staircase=True)
net = tflearn.regression(net, optimizer=mom,
                         loss='categorical_crossentropy')

# Training
model = tflearn.DNN(net, checkpoint_path='../../../../weights/checkpoints/planet_cloudless/resnet_planet_cloudless',
                    max_checkpoints=1, tensorboard_verbose=3)

#model.load('../../checkpoints/resnet_sat6-4500')
model.fit(X_train, y_train, n_epoch=5, validation_set=(X_test, y_test),
          snapshot_epoch=False, snapshot_step=1000,
          show_metric=True, batch_size=256, shuffle=True,
          run_id='resnet_planet_cloudless')

print(model.evaluate(X_test, y_test, batch_size=512))

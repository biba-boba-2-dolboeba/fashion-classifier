from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow и tf.keras
import tensorflow as tf
from tensorflow import keras

# Вспомогательные библиотеки
import math
from PIL import Image as im
import cv2
import numpy as np
import matplotlib.pyplot as plt


from tensorflow.contrib.learn.python.learn.datasets.mnist import extract_images, extract_labels


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


#метод переводит в негатив переданное изображение
def load_img(imgpath):
    img = cv2.imread(imgpath, 0)
    plt.imshow(img, cmap=plt.cm.binary)
    plt.xticks([])
    plt.yticks([])
    plt.savefig("present_classification_img.png",  bbox_inches='tight')
    plt.show()




load_img('pathToimg')



image = keras.preprocessing.image.load_img('present_classification_img.png',
                                           color_mode='grayscale',
                                           target_size=(28,28),
                                           interpolation='box')

array_image = keras.preprocessing.image.img_to_array(image)
array_image = np.expand_dims(array_image, axis=0) / 255.0
array_image = np.squeeze(array_image, 3)


print(array_image.shape)


model = keras.models.load_model('model.h5')


prediction = model.predict(array_image)


print(prediction[0])

print(max(prediction[0]))

print(np.argmax(prediction[0]))

print('предположение сети '+class_names[np.argmax(prediction[0])])



import matplotlib.pyplot as plt
import numpy as np
import cv2
import tensorflow as tf

from tensorflow import keras
from keras import layers
from keras.models import Sequential
import pathlib

def getNames () :
    classNameFile = open("names.txt","r")
    names = classNameFile.readlines()
    for i in range(len(names)) :
        names[i] = str(names[i]).replace('\n','')
    return names

model = tf.keras.models.load_model('my_model')

# Check its architecture
# model.summary()

class_names = getNames()


inputPath = 'sunflower.jpg'

im = cv2.imread(inputPath)


img_height = 180
img_width = 180


img = tf.keras.utils.load_img(
    inputPath, target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

# print(model._predict_counter)


# predictions = model.predict(img_array)

# model.

predictions = model.predict(img_array)

# print(predictions)

score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)

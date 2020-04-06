import numpy

import argparse
import os, sys
from PIL import Image, ImageDraw
import tensorflow as tf
from tensorflow.keras import backend
from tensorflow import keras
from image_handler import produce_crops
from keras.preprocessing.image import ImageDataGenerator
import face_recognition_models
BATCH_SIZE = 32
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
model = keras.Sequential()

model.add(keras.layers.Conv2D(32, batch_input_shape=(BATCH_SIZE, IMAGE_WIDTH, IMAGE_HEIGHT, 3), kernel_size=(5, 5),
                              strides=(1, 1),
                              activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Conv2D(128, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Conv2D(256, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(500, activation='relu'))
model.add(keras.layers.Dense(250, activation='relu'))
model.add(keras.layers.Dense(50, activation='relu'))

model.add(keras.layers.Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
train_gen = ImageDataGenerator(rescale=1. / 255)
validation_gen = ImageDataGenerator(rescale=1. / 255)

train_dataset = train_gen.flow_from_directory("dataset/train/", shuffle=True, batch_size=BATCH_SIZE,
                                              target_size=(IMAGE_WIDTH, IMAGE_HEIGHT))

validation_dataset = validation_gen.flow_from_directory("dataset/validation", shuffle=True, batch_size=BATCH_SIZE,
                                                        target_size=(IMAGE_WIDTH, IMAGE_HEIGHT))
if not os.path.exists("model.h5"):
    history = model.fit(train_dataset, epochs=10, verbose=2,
                        shuffle=True, class_weight={0: 1, 1: 20}, validation_data=validation_dataset)
    model.save("model.h5")
else:
    model.load_weights("model.h5")
img = Image.open("waldoface.jpg")
draw = ImageDraw.ImageDraw(img)
max_value, it, jt = 0, 0, 0
for i, j, crop in produce_crops("waldoface.jpg", size=(IMAGE_WIDTH, IMAGE_HEIGHT)):
    x_t = numpy.array(crop).reshape((1, IMAGE_WIDTH, IMAGE_HEIGHT, 3)) / 255
    prediction = model.predict(x_t)
    index = numpy.argmax(prediction[0])
    print(prediction[0])
    if prediction[0][1] > max_value:
        max_value = prediction[0][1]
        it = i
        jt = j
print("waldo at ", it, jt)
draw.rectangle([(it * IMAGE_WIDTH, jt * IMAGE_HEIGHT), ((it + 1) * IMAGE_WIDTH, (jt + 1) * IMAGE_HEIGHT)],
               outline='blue')

img.show()

import pandas as pd
import numpy as np
from keras.models import Sequential

from keras.layers import Dense


from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten


model=Sequential()


model.add(Convolution2D(filters=32,kernel_size=(3,3),activation='relu',input_shape=(64,64,3)))


model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Flatten())


model.add(Dense(units=128,activation='relu'))


model.add(Dense(units=64,activation='relu'))


model.add(Dense(units=32,activation='relu'))
model.add(Dense(units=16,activation='relu'))
model.add(Dense(units=8,activation='relu'))
model.add(Dense(units=1,activation='sigmoid'))


print(model.summary())


model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


from keras_preprocessing.image import ImageDataGenerator


train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
training_set = train_datagen.flow_from_directory(
        '/model_files/images/images/train/',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

model.fit(training_set,
        steps_per_epoch=100,
        epochs=10,
        validation_data=test_set,
        validation_steps=800)


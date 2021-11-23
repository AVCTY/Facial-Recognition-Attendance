import tensorflow as tf
from tensorflow import keras
import process_data as p_data
import numpy as np

model = tf.keras.Sequential([
    #tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# do training part first

# get classes
labels = p_data.process_full_labels("train_data")
distinct_labels = p_data.process_distinct_labels("train_data")

# read images
training_images = p_data.read_images("train_data", distinct_labels)

# need to nparray again to train
np_images = np.array( training_images )
print ( np_images.shape ) # unflatten gives something like (28, 250, 250, 3)
#np_images = np_images.reshape(-1, 28*28)
#np_images = np_images.astype('float32') / 255
#np_images = np_images.flatten()


int_labels = labels_str_to_int( labels )
np_labels = np.array( int_labels, dtype="uint8" ) # this part is causing problems, the "n"
print ( np_labels.shape )
#np_labels = np_labels.reshape(-1, 28*28)

# fit model
#model.fit(np_images, np_labels, epochs=1)
model.fit(np_images, np_labels, epochs=1)

# save model
# https://www.tensorflow.org/tutorials/keras/save_and_load
filename = 'saved_cnn'

model.save('saved_model/cnn_model')
model = tf.keras.models.load_model('saved_model/cnn_model')

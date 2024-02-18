import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
from functools import partial
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Normalization
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers.experimental import preprocessing
import tensorflow as tf

class CustomActivation(tf.keras.layers.Layer):
  def __init__(self):
    super(CustomActivation,self).__init__()
    def call(self, inputs):
      return tf.where(x < 1, tf.ones_like(x), x)

def custom_loss(y_true, y_pred):
    # Define weights for each output
    weights = tf.constant([1.0, 1.0, 0.5])  # Higher weight for the first two outputs

    # Compute mean squared error with weighted difference
    error = tf.square(y_true - y_pred) * weights
    return tf.reduce_mean(error, axis=-1)  # Mean over all outputs

def custom_activation(x):
    return tf.where(x < 1, tf.ones_like(x), x)

data_input = np.loadtxt('input_normalized.csv', delimiter=',')
data_output = np.loadtxt('rloss_normalized50.csv', delimiter=',')

normalizer_input = preprocessing.Normalization()
normalizer_input.adapt(data_input)
X_normalized = normalizer_input(data_input)
print(X_normalized)
X_array = X_normalized.numpy()
#sementara, coba 70% data train data 30% data test
X_train, X_test, y_train, y_test = train_test_split(X_array, data_output, test_size=0.3, random_state=42)
#norm_layer = Normalization(input_shape=X_train.shape[1:])
#lets try without normalization
model = Sequential([Flatten(input_shape=(6,)),
                    Dense(30, activation='relu'),
                    Dense(25, activation='relu'),
                    Dense(25, activation='relu'),
                    Dense(10, activation='relu'),
                    Dense(1, activation=CustomActivation())])

optimizer = Adam(learning_rate=1e-4)
model.compile(loss="mse", optimizer=optimizer, metrics=["mean_absolute_percentage_error"])
#norm_layer.adapt(X_train)

history = model.fit(X_train, y_train, epochs=5000, validation_data = (X_test, y_test))

#kpoint_path = "/content/drive/MyDrive/simulasi/cyclotron/"
model.save('cyclotron_rloss50_sixinput.keras')
#model.save('/content/drive/MyDrive/simulasi/cyclotron/cyclotron_simple_sixinput.h5')

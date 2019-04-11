import tensorflow as tf
import numpy as np
from tensorflow import keras

# TF v 1.13.1
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')
xs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
ys = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5])
model.fit(xs, ys, epochs=5000)
print(model.predict([21.0]))


# 11.0  > 6.028263   - 500   - 10
# 11.0  > 6.0056796  - 1000  - 10
# 11.0  > 6.000003   - 5000  - 10 <<
# 11.0  > 6.000003   - 10000 - 10
# 21.0  > 11.000009  - 5000  - 10 <<
# 21.0  > 11.000009  - 10000 - 10
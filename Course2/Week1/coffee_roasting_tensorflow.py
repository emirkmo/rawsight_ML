import tensorflow as tf
from datasets import load_coffee_data
import numpy as np

data = load_coffee_data()

normalizer_tf = tf.keras.layers.Normalization(axis=-1)
normalizer_tf.adapt(data.X_train)  # Learn mean and variance
norm_X_tf = normalizer_tf(data.X_train)  # Apply learned mean and variance
print(f"Temperature Max, Min post normalization: {np.max(norm_X_tf[:,0]):0.2f}, {np.min(norm_X_tf[:,0]):0.2f}")
print(f"Duration    Max, Min post normalization: {np.max(norm_X_tf[:,1]):0.2f}, {np.min(norm_X_tf[:,1]):0.2f}")

data.normalize_features()
print(f"Temperature Max, Min post normalization: {np.max(data.X_train[:,0]):0.2f}, {np.min(data.X_train[:,0]):0.2f}")
print(f"Duration    Max, Min post normalization: {np.max(data.X_train[:,1]):0.2f}, {np.min(data.X_train[:,1]):0.2f}")

data.X_train = np.tile(data.X_train, (1000, 1))
data.y_train = np.tile(data.y_train.reshape(-1, 1), (1000, 1))

tf.random.set_seed(1234)  # applied to achieve consistent results
model = tf.keras.Sequential(
    [
        tf.keras.Input(shape=(2,)),
        tf.keras.layers.Dense(3, activation='sigmoid', name='layer1'),
        tf.keras.layers.Dense(1, activation='sigmoid', name='layer2')
    ]
)

model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              )
model.fit(data.X_train, data.y_train, epochs=10,)


data.X_test = data.normalizer.apply_norm(np.array([
    [200, 13.9],  # postive example
    [200, 17]]))   # negative example
predictions = model.predict(data.X_test)
decisions = (predictions >= 0.5).astype(int)
print(f"decisions = \n{decisions}")

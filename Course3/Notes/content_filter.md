# Content based filtering

## Difference to collaborative filtering

Learning to match features instead of learning
from parameters on features.

So users have features, movies have features,
create vector for each feature set, predict user/movie
rating match. (Recommend movie to user or predict user score for movie).

No constant vector `b`.

`V_M . V_U`. Must calculate from feature vector.

### How to calculate V? Use deep learning (neural network NN)

NN output layer should not have single unit, but many
(unit per vector element) HOW MANY?? (idk, 32). Hidden layers can be any complexity, but output layers of `V_M`` and `V_U` must match!

Instead of dot product, simply take sigmoid etc. of
V_U and V_M, and find where g(V.V) = 1.

## Cost Function

```Latex
J = Sum (v_u(j) . v_m(i) - y(i,j)) + NN regularization.
```

Basically need labels Y, with existing movie/user ratings(matches).
Same cost function for NN for both vectors.

### Tips

To Find: Similar movies take L2 norm distance.
This can and should be pre-computed!
Now you have a similarity matrix. Movies are related like
a graph.

NN benefit realized: Allows easily integrating movie and
user NN by taking dot product of outer layer of each.
Really powerful!

The feature engineering is critical.

Algorithm as described is computational expensive to run,
need modifications to scale.

## Scale up Recommender system

Retrieval & Ranking

### Retrieval

Generate large list of plausible item candidates.

Use pre-computed `||Vm(k) - V_m(j) || ^2`

Find similar movies, most viewed 3 genres, top movies of
all times, top X movies in same country, etc.

### Ranking

Now we have small list of movies, rank them.
V_m can be pre-computed (since new users and user
feature values change way more often).
We only need to calculate V_u from pared retrieval
step, which is fast. Can be done on edge.

Retrieval step should be tuned using offline experiments
and A/B testing, etc.

## Ethics

Don't be evil. Don't be naive.
Think about goal. Think about bad actors.

Be transparent with users. Need to be careful with exploitative recommendations.

## Tensorflow Recommender Algorithm

Same as NN, Sequential model from keras

```Python
import tensorflow as tf

user_nn = tf.keras.models.Sequential([tf.keras.layers.Dense(..., activation='relu'), ...])
...

# add input layer
user_input = tf.keras.layers.Input(shape=(num_user_features))

vu = user_nn(input_user)
vu = tf.linalg.l2_normalize(vu, axis=1) # normalize the L2 norm, Yo!
# Repeat for item/movie
vm = ...

# Keras dot product layer
tf.keras.layers.Dot(axes=1)([vu, vm])

# Use simple MSE for loss
cost_fn = tf.keras.losses.MeanSquaredError # I guess, idk.
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

# Training model using keras api.
n_iterations = 30
model = tf.keras.Model([input_user, input_item], output)
model.compile(optimizer=optimizer, loss=cost_fn)
model.fit([user_train, item_train], y_train, epochs=n_iterations)
```

### Lab

Using sklearn StandardScaler for user but MinMaxScaler for target. Not clear why. Uses `inverse_transform` of scaler to get back originals. Ready-made `test_train_split` for the split with a 20% test.

Based on the fact that test loss is similar to training
loss, we infer that model has not substantially overfit.
(Weird to not use CV set, but model params and parts were
just given, so no need.)

#%%
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

train, test= tfds.load(
    'rock_paper_scissors',
    split=['train','test'],
)

train_numpy = np.vstack(tfds.as_numpy(train))
test_numpy = np.vstack(tfds.as_numpy(test))

train_numpy = train_numpy[:int(train_numpy.size*0.8)]
test_numpy = test_numpy[:int(test_numpy.size*0.8)]

X_train = np.array(list(map(lambda x: x[0]['image'], train_numpy)))
Y_train = np.array(list(map(lambda x: x[0]['label'], train_numpy)))
X_train = X_train.transpose(0, 3, 1, 2) / 255.0
Y_train = Y_train.flatten()

X_test = np.array(list(map(lambda x: x[0]['image'], test_numpy)))
Y_test = np.array(list(map(lambda x: x[0]['label'], test_numpy)))
X_test = X_test.transpose(0, 3, 1, 2) / 255.0
Y_test = Y_test.flatten()

# %%
import tensorflow as tf
#.load_data() by default returns a split between training and test set. 
# We then adjust the training set into a format that can be accepted by our CNN
(X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.cifar10.load_data()
X_train = X_train.transpose(0, 3, 1, 2) / 255.0
Y_train = Y_train.flatten()
X_test = X_test.transpose(0, 3, 1, 2) / 255.0
Y_test = Y_test.flatten()


# %%

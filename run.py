
# Used resource:
#  https://stackoverflow.com/questions/40427435/extract-images-from-idx3-ubyte-file-or-gzip-via-python


import gzip
import tensorflow as tf
import numpy as np

image_size = 28
train_images = 60000
test_images = 10000

train_images_path = 'dtset/train-images-idx3-ubyte.gz'
train_labels_path = 'dtset/train-labels-idx1-ubyte.gz'
test_images_path = 'dtset/t10k-images-idx3-ubyte.gz'
test_labels_path = 'dtset/t10k-labels-idx1-ubyte.gz'
checkpoint_path = 'checkpoints/model.ckpt'

"""
import matplotlib.pyplot as plt
image = np.asarray(data[0]).squeeze()
plt.imshow(image)
plt.show()
"""

##

def load_dataset(datapath, labelpath, size):
    f_data = gzip.open(datapath, 'r')
    f_labels = gzip.open(labelpath, 'r')

    # offset
    f_data.read(16)
    f_labels.read(8)

    # images
    buf = f_data.read(image_size * image_size * size)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    data = data.reshape(size, image_size, image_size)

    # labels
    buf = f_labels.read(image_size * image_size * size)
    labels = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)

    data = data / 255.0

    return data, labels

def train():
    train_imgs, train_labels = load_dataset(train_images_path, train_labels_path, train_images)
    test_imgs, test_labels = load_dataset(test_images_path, test_labels_path, test_images)

    # Step 1: Define model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
    ])

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['accuracy'])
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
    model.fit(train_imgs, train_labels, epochs=20, callbacks=[cp_callback])
    

if __name__ == '__main__':
    train()
    # extract()

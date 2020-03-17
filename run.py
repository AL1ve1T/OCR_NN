
# Used resource:
#  https://stackoverflow.com/questions/40427435/extract-images-from-idx3-ubyte-file-or-gzip-via-python


import gzip
import tensorflow as tf
import numpy as np

image_size = 28
train_images = 5
test_images = 5

train_images_path = 'dtset/train-images-idx3-ubyte.gz'
train_labels_path = ''
test_images_path = ''
test_labels_path = ''

buf = f.read(image_size * image_size * num_images)
data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
data = data.reshape(num_images, image_size, image_size, 1)

print(len(data[4][0]))

"""
import matplotlib.pyplot as plt
image = np.asarray(data[0]).squeeze()
plt.imshow(image)
plt.show()
"""

##

def load_dataset(datapath, labelpath):
    f_data = gzip.open(datapath, 'r')
    f_labels = gzip.open(labelpath, 'r')

    # offset
    f_data.read(16)
    f_labels.read(8)

    # images
    buf = f_data.read(image_size * image_size * num_images)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    data = data.reshape(num_images, image_size, image_size, 1)

    # labels
    buf = f_labels.read(image_size * image_size * num_images)
    labels = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    labels = labels.reshape(num_images, image_size, image_size, 1)

    return data, labels

def build_graph():
    train_imgs, train_labels = load_dataset(train_images_path, train_labels_path)
    test_imgs, test_labels = load_dataset(test_images_path, test_labels_path)

    
    

if __name__ == '__main__':
    load_dataset()
    build_graph()
      train()
      test()
    extract()

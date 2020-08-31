import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa
from scipy.signal import hamming
from scipy.signal import lfilter, butter
import decimal
import math
from scipy.io import wavfile
import collections
import os
import glob

def create_model(input_shape, num_classes):
    i = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(96, (7,7), strides=(2,2), padding="valid", kernel_regularizer=tf.keras.regularizers.L2(5e-4), name="conv1")(i)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D((3,3), strides=(2,2), name="mpool1")(x)
    x = tf.keras.layers.Conv2D(256, (5,5), strides=(2,2), padding="valid", kernel_regularizer=tf.keras.regularizers.L2(5e-4), name="conv2")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D((3,3), strides=(2,2), name="mpool2")(x)
    x = tf.keras.layers.Conv2D(384, (3,3), strides=(1,1), padding="same", kernel_regularizer=tf.keras.regularizers.L2(5e-4), name="conv3")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(256, (3,3), strides=(1,1), padding="same", kernel_regularizer=tf.keras.regularizers.L2(5e-4), name="conv4")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(256, (3,3), strides=(1,1), padding="same", kernel_regularizer=tf.keras.regularizers.L2(5e-4), name="conv5")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D((5,3), strides=(3,2), name="mpool5")(x)
    x = tf.keras.layers.Conv2D(4096, (9,1), strides=1, padding="valid", kernel_regularizer=tf.keras.regularizers.L2(5e-4), name="fc6")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.math.reduce_mean(x, axis=[1,2], name="apool6")
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1024, kernel_regularizer=tf.keras.regularizers.L2(5e-4), name="fc7")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dense(num_classes, kernel_regularizer=tf.keras.regularizers.L2(5e-4), name="logits")(x)
    
    model = tf.keras.Model(inputs=i, outputs=x)

    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1e-2, momentum=0.9),
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.TopKCategoricalAccuracy()])

    return model

def calculate_splits(extract_path):
    iden_splits_path = "../iden_split.txt"
    data_splits = collections.defaultdict(set)
    with open(iden_splits_path, "r") as f:
        for line in f:
            group, path = line.strip().split()
            path = path[:-4] + ".npx"
            full_path = os.path.join(extract_path, path)
            if not os.path.isfile(full_path):
                continue
            split_name = {1: 'train', 2: 'validation', 3: 'test'}[int(group)]
            data_splits[split_name].add(full_path.strip())
    return data_splits

def generate_example(filename_tf, num_labels_tf, split_tf):
    filename = filename_tf.numpy().decode()
    num_labels = num_labels_tf.numpy()
    split = split_tf.numpy()
    speaker = filename.split("/")[-3]
    label = int(speaker[3:]) - 1 # there is no label 0
    one_hot_label = np.zeros(num_labels)
    one_hot_label[label] = 1
    with open(filename, "rb") as f:
        stft = np.load(f).astype(np.float32)
    if stft.shape[1] == 300:
        pass
    elif split == b"train":
        start = np.random.randint(0, stft.shape[1] - 300)
        stft = stft[:, start:start+300]
    stft = np.expand_dims(stft, -1)
    return stft, one_hot_label

def load(filename, num_labels, input_shape, split):
    [stft, label] = tf.py_function(func=generate_example, inp=[filename, num_labels, split], Tout=[tf.float32, tf.int16])
    stft.set_shape(input_shape)
    label.set_shape([num_labels])
    return stft, label

def input_fn(filenames, split, batch_size, input_shape, num_labels):
    d = tf.data.Dataset.list_files(filenames)
    if split == "train":
        d = d.shuffle(buffer_size=300)
        d = d.map(lambda filename: load(filename, num_labels, input_shape, split), num_parallel_calls=16, deterministic=False)
        d = d.batch(batch_size=batch_size, drop_remainder=True)
    elif split == "test":
        d = d.map(lambda filename: load(filename, num_labels, input_shape, split), num_parallel_calls=16)
    return d

num_classes = 1251
batch_size = 100
input_shape = [512, None, 1]

save_path = "./voxceleb-cnn"
wav_path = "./voxceleb_train/wav"

splits = calculate_splits(wav_path)
train_filenames = list(splits["train"]) + list(splits["validation"])
test_filenames = list(splits["test"])
train_set = input_fn(train_filenames, "train", batch_size, input_shape, num_classes)
test_set = input_fn(test_filenames, "test", -1, input_shape, num_classes) # cannot batch, since dimensions vary between samples

model = create_model(input_shape, num_classes)
print(model.summary())

# lr scheduler parameter
gamma = 10 ** (np.log10(1e-4 / 1e-2) / (30 - 1))
def scheduler(epoch, lr):
    if epoch > 0:
        print("Learning rate is now", lr * gamma)
        return lr * gamma  
    else:
        return lr
lrs = tf.keras.callbacks.LearningRateScheduler(scheduler)

history_callback = model.fit(train_set, epochs=10, callbacks=[lrs])
test_acc = tf.keras.metrics.CategoricalAccuracy()
test_acc_top_k = tf.keras.metrics.TopKCategoricalAccuracy(k=5)
for stft, label in test_set:
    pred = model.predict(np.expand_dims(stft, axis=0))
    label = np.expand_dims(label, axis=0)
    test_acc.update_state(label, pred)
    test_acc_top_k.update_state(label, pred)
test_acc_result = test_acc.result().numpy()
print("Test_acc result: ", test_acc_result)
print("Test_acc_top_5 result: ", test_acc_top_k.result().numpy())

model.save(save_path)

# -*- coding: utf-8 -*-
"""implementing-resnet-18-using-keras.ipynb

Original file is located at
    https://colab.research.google.com/drive/1nNtKwEhRR7PtF8gUqu0UET93ZcV_Aw5Y

# Implementing ResNet-18 Using Keras

## preprocess
"""

#Import Libraries
import numpy as np
import tensorflow as tf
import keras
from tensorflow.keras import datasets,models,layers
from keras import backend as K
from tensorflow.keras.applications import ResNet50, ResNet101
import os

use_salr = (True if os.environ["use_salr"] == "True" else False) if "use_salr" in os.environ else True

device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  print(
      '\n\nThis error most likely means that this notebook is not '
      'configured to use a GPU.  Change this in Notebook Settings via the '
      'command palette (cmd/ctrl-shift-P) or the Edit menu.\n\n')
#   raise SystemError('GPU device not found')

"""# Implement ResNet-18 model

Codes below are taken from my [Github](https://github.com/songrise/CNN_Keras)
"""

"""
ResNet-18
Reference:
[1] K. He et al. Deep Residual Learning for Image Recognition. CVPR, 2016
[2] K. He, X. Zhang, S. Ren, and J. Sun. Delving deep into rectifiers:
Surpassing human-level performance on imagenet classification. In
ICCV, 2015.
"""


from keras.callbacks import EarlyStopping
from keras.layers import Dense, Conv2D,  MaxPool2D, Flatten, GlobalAveragePooling2D,  BatchNormalization, Layer, Add
from keras.models import Sequential
from keras.models import Model
import tensorflow as tf
from keras.datasets import cifar10
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD

total_batch_size = 100
loss_fn = keras.losses.CategoricalCrossentropy()

ascent_descent_learning_rate = 0.002
n_ascent = 5
n_descent = 5

base_learning_rate = 0.01

class ResnetBlock(Model):
    """
    A standard resnet block.
    """

    def __init__(self, channels: int, down_sample=False):
        """
        channels: same as number of convolution kernels
        """
        super().__init__()

        self.__channels = channels
        self.__down_sample = down_sample
        self.__strides = [2, 1] if down_sample else [1, 1]

        KERNEL_SIZE = (3, 3)
        # use He initialization, instead of Xavier (a.k.a 'glorot_uniform' in Keras), as suggested in [2]
        INIT_SCHEME = "he_normal"

        self.conv_1 = Conv2D(self.__channels, strides=self.__strides[0],
                             kernel_size=KERNEL_SIZE, padding="same", kernel_initializer=INIT_SCHEME)
        self.bn_1 = BatchNormalization()
        self.conv_2 = Conv2D(self.__channels, strides=self.__strides[1],
                             kernel_size=KERNEL_SIZE, padding="same", kernel_initializer=INIT_SCHEME)
        self.bn_2 = BatchNormalization()
        self.merge = Add()

        if self.__down_sample:
            # perform down sampling using stride of 2, according to [1].
            self.res_conv = Conv2D(
                self.__channels, strides=2, kernel_size=(1, 1), kernel_initializer=INIT_SCHEME, padding="same")
            self.res_bn = BatchNormalization()

    def call(self, inputs):
        res = inputs

        x = self.conv_1(inputs)
        x = self.bn_1(x)
        x = tf.nn.relu(x)
        x = self.conv_2(x)
        x = self.bn_2(x)

        if self.__down_sample:
            res = self.res_conv(res)
            res = self.res_bn(res)

        # if not perform down sample, then add a shortcut directly
        x = self.merge([x, res])
        out = tf.nn.relu(x)
        return out


class ResNet18(Model):

    def __init__(self, num_classes, **kwargs):
        """
            num_classes: number of classes in specific classification task.
        """
        super().__init__(**kwargs)
        self.conv_1 = Conv2D(64, (7, 7), strides=2,
                             padding="same", kernel_initializer="he_normal")
        self.init_bn = BatchNormalization()
        self.pool_2 = MaxPool2D(pool_size=(2, 2), strides=2, padding="same")
        self.res_1_1 = ResnetBlock(64)
        self.res_1_2 = ResnetBlock(64)
        self.res_2_1 = ResnetBlock(128, down_sample=True)
        self.res_2_2 = ResnetBlock(128)
        self.res_3_1 = ResnetBlock(256, down_sample=True)
        self.res_3_2 = ResnetBlock(256)
        self.res_4_1 = ResnetBlock(512, down_sample=True)
        self.res_4_2 = ResnetBlock(512)
        self.avg_pool = GlobalAveragePooling2D()
        self.flat = Flatten()
        self.fc = Dense(num_classes, activation="softmax")

    def call(self, inputs):
        out = self.conv_1(inputs)
        out = self.init_bn(out)
        out = tf.nn.relu(out)
        out = self.pool_2(out)
        for res_block in [self.res_1_1, self.res_1_2, self.res_2_1, self.res_2_2, self.res_3_1, self.res_3_2, self.res_4_1, self.res_4_2]:
            out = res_block(out)
        out = self.avg_pool(out)
        out = self.flat(out)
        out = self.fc(out)
        return out

cluster_resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()
os.environ["GRPC_FAIL_FAST"] = "use_caller"
server = tf.distribute.Server(
    cluster_resolver.cluster_spec(),
    job_name=cluster_resolver.task_type,
    task_index=cluster_resolver.task_id,
    protocol=cluster_resolver.rpc_layer or "grpc",
    start=True)

variable_partitioner = (
    tf.distribute.experimental.partitioners.MinSizePartitioner(
        min_shard_bytes=(256 << 10),
        max_shards=1))

strategy = tf.distribute.experimental.ParameterServerStrategy(
        cluster_resolver,
        variable_partitioner=variable_partitioner)

def dataset_fn(input_context):
    global_batch_size = total_batch_size
    batch_size = input_context.get_per_replica_batch_size(global_batch_size)
    
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

    # Normalize the data.
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255.0
    X_test /= 255.0
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.2,shuffle = True)


    encoder = OneHotEncoder()
    encoder.fit(Y_train)
    Y_train = encoder.transform(Y_train).toarray()
    Y_test = encoder.transform(Y_test).toarray()
    Y_val =  encoder.transform(Y_val).toarray()

    """## data augmentation"""
    aug = ImageDataGenerator(horizontal_flip=True, width_shift_range=0.05,
                                height_shift_range=0.05)
    aug.fit(X_train)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    train_dataset = train_dataset.batch(batch_size)
    return train_dataset

dc = tf.keras.utils.experimental.DatasetCreator(dataset_fn)

with strategy.scope():
    model = ResNet18(10)
    model.build(input_shape = (None,32,32,3))
    #use categorical_crossentropy since the label is one-hot encoded

    optimizer = SGD(learning_rate=base_learning_rate,momentum=0.9)

    stochastic_sharpness_list = np.array([])

@tf.function(autograph=False)
def step_fn(iterator):

    def replica_fn(batch_data, labels):
        with tf.GradientTape() as tape:

            # Run the forward pass of the layer.
            # The operations that the layer applies
            # to its inputs are going to be recorded
            # on the GradientTape.
            logits = model(batch_data, training=True)  # Logits for this minibatch

            # Compute the loss value for this minibatch.
            loss_value = loss_fn(labels, logits)

        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(loss_value, model.trainable_weights)

        if cluster_resolver.task_id == 0:
            pass
        ################
        ### FOR SALR ###
        ################
        if  cluster_resolver.task_id == 0 and use_salr:
            model_ascent = ResNet18(10)
            model_ascent.build(input_shape = (None,32,32,3))
            model_ascent.set_weights(model.get_weights())
            model_descent = ResNet18(10)
            model_descent.build(input_shape = (None,32,32,3))
            model_descent.set_weights(model.get_weights())

            optimizer_ascent = SGD(learning_rate=-ascent_descent_learning_rate, momentum=0.9,  clipnorm=1.0)
            optimizer_descent = SGD(learning_rate=ascent_descent_learning_rate, momentum=0.9,  clipnorm=1.0)

            for i in range(n_ascent):
                with tf.GradientTape() as tape_ascent:

                    # Run the forward pass of the layer.
                    # The operations that the layer applies
                    # to its inputs are going to be recorded
                    # on the GradientTape.
                    logits_ascent = model_ascent(batch_data, training=True)  # Logits for this minibatch

                    # Compute the loss value for this minibatch.
                    loss_value_ascent = loss_fn(labels, logits_ascent)

                grads_ascent = tape_ascent.gradient(loss_value_ascent, model_ascent.trainable_weights)
                clipped_grads_ascent, _ = tf.clip_by_global_norm(grads_ascent, 1.0)
                optimizer_ascent.apply_gradients(zip(clipped_grads_ascent, model_ascent.trainable_weights))
            
            for i in range(n_descent):
                with tf.GradientTape() as tape_descent:

                    # Run the forward pass of the layer.
                    # The operations that the layer applies
                    # to its inputs are going to be recorded
                    # on the GradientTape.
                    logits_descent = model_descent(batch_data, training=True)  # Logits for this minibatch

                    # Compute the loss value for this minibatch.
                    loss_value_descent = loss_fn(labels, logits_descent)
                
                grads_descent = tape_descent.gradient(loss_value_descent, model_descent.trainable_weights)
                clipped_grads_descent, _ = tf.clip_by_global_norm(grads_descent, 1.0)
                optimizer_descent.apply_gradients(zip(clipped_grads_descent, model_descent.trainable_weights))
            
            stochastic_sharpness = loss_value_ascent - loss_value_descent
            print("\n   Stochastic Sharpness %3.10f" % (stochastic_sharpness,))
            stochastic_sharpness_list =  np.append(stochastic_sharpness_list, stochastic_sharpness)
            median_sharpness = np.median(stochastic_sharpness_list)
            
            print("\n   Median Sharpness %3.10f" % (median_sharpness,))

            current_lr = K.get_value(optimizer.lr)
            K.set_value(optimizer.lr, base_learning_rate * stochastic_sharpness / median_sharpness)
            print("\n   Learning Rate %3.10f ==> %3.10f" % (current_lr, K.get_value(optimizer.lr),))


            ################
            ### END SALR ###
            ################

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        # Log every 200 batches
        print(
                "Training loss (for one batch): %.4f"  % (float(loss_value))
        )
        return loss_value

    batch_data, labels = next(iterator)
    losses = strategy.run(replica_fn, args=(batch_data, labels))
    return strategy.reduce(tf.distribute.ReduceOp.SUM, losses, axis=None)


coordinator = tf.distribute.experimental.coordinator.ClusterCoordinator(strategy)
# Commented out IPython magic to ensure Python compatibility.
#  with tf.device('/device:GPU:0'):

@tf.function
def per_worker_dataset_fn():
    return strategy.distribute_datasets_from_function(dataset_fn)

per_worker_dataset = coordinator.create_per_worker_dataset(per_worker_dataset_fn)
per_worker_iterator = iter(per_worker_dataset)

num_epoches = 5
steps_per_epoch = 400
for i in range(num_epoches):
    for _ in range(steps_per_epoch):
        coordinator.schedule(step_fn, args=(per_worker_iterator,))
    # Wait at epoch boundaries.
    while not coordinator.done():
        print ("Finished epoch %d." % i)

    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

    # Normalize the data.
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255.0
    X_test /= 255.0
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.2,shuffle = True)


    encoder = OneHotEncoder()
    encoder.fit(Y_train)
    Y_train = encoder.transform(Y_train).toarray()
    Y_test = encoder.transform(Y_test).toarray()
    Y_val =  encoder.transform(Y_val).toarray()

    PREDICTED_CLASSES = model.predict(X_test)
    print(PREDICTED_CLASSES)
    print(Y_test)
    true=0
    for i in range(len(Y_test)):
        if (tf.argmax(PREDICTED_CLASSES[i]) == tf.argmax(Y_test[i])):
            true+=1
    acc_test = true/len(PREDICTED_CLASSES)

    PREDICTED_CLASSES = model.predict(X_train)
    print(PREDICTED_CLASSES)
    print(Y_train)
    true=0
    for i in range(len(Y_train)):
        if (tf.argmax(PREDICTED_CLASSES[i]) == tf.argmax(Y_train[i])):
            true+=1
    acc_train = true/len(PREDICTED_CLASSES)


    print("training accuracy: %3.10f" % acc_train)
    print("testing accuracy : %3.10f" % acc_test)
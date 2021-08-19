'''
The code is inspired from François Chollet's answer to the following quora question[1] and distributed tensorflow tutorial[2].
It runs the Keras MNIST mlp example across multiple servers.
This sample code runs multiple processes on a single host. It can be configured 
to run on multiple hosts simply by chaning the host names given in *ClusterSpec*.
Training the model:
Start the parameter server
  python keras_distributed.py --job_name="ps" --task_index=0
  
Start the three workers
  python keras_distributed.py --job_name="worker" --task_index=0
  python keras_distributed.py --job_name="worker" --task_index=1
  python keras_distributed.py --job_name="worker" --task_index=2
  
[1] https://www.quora.com/What-is-the-state-of-distributed-learning-multi-GPU-and-across-multiple-hosts-in-Keras-and-what-are-the-future-plans
[2] https://www.tensorflow.org/deploy/distributed
'''

from keras.layers import Dense, Conv2D,  MaxPool2D, Flatten, GlobalAveragePooling2D,  BatchNormalization, Layer, Add
from keras.models import Sequential
from keras.models import Model
import tensorflow as tf
import keras
from keras.datasets import cifar10
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
import numpy as np
import os
import ast

batch_size = 100

ascent_descent_learning_rate = 0.002
n_ascent = 5
n_descent = 5

base_learning_rate = 0.01
epochs = 40

log_frequency = 200

class Empty:
  pass

FLAGS = Empty()

TF_CONFIG = ast.literal_eval(os.environ["TF_CONFIG"])
FLAGS.job_name = TF_CONFIG["task"]["type"]
FLAGS.task_index = TF_CONFIG["task"]["index"]
FLAGS.ps_hosts = ",".join(TF_CONFIG["cluster"]["ps"])
FLAGS.worker_hosts = ",".join(TF_CONFIG["cluster"]["worker"])
FLAGS.use_salr = (True if os.environ["use_salr"] == "True" else False) if "use_salr" in os.environ else True

ps_hosts = FLAGS.ps_hosts.split(",")
worker_hosts = FLAGS.worker_hosts.split(",")

cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

# start a server for a specific task
server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)


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
                             kernel_size=KERNEL_SIZE, padding="same", kernel_initializer=INIT_SCHEME, input_shape=(None,32,32,3))
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
                             padding="same", kernel_initializer="he_normal", input_shape=(None,32,32,3))
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
    
    def build(self, input_shape):
        self.fc(self.flat(self.avg_pool(self.res_4_2(self.res_4_1(self.res_3_2(self.res_3_1(self.res_2_2(self.res_2_1(self.res_1_2(
            self.res_1_1(self.pool_2(tf.nn.relu(self.init_bn(self.conv_1(tf.keras.layers.Input(shape=input_shape, name="input_x"))))))))))))))))

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



if FLAGS.job_name == "ps":
    server.join()

elif FLAGS.job_name == "worker":
    # Assign operations to local server
    with tf.device(tf.train.replica_device_setter(
            worker_device="/job:worker/task:%d" % FLAGS.task_index,
            cluster=cluster)):
        # Adding TF Cifar10 Data ..
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

        aug = ImageDataGenerator(horizontal_flip=True, width_shift_range=0.05,
                                    height_shift_range=0.05)
        aug.fit(X_train)

        keras.backend.set_learning_phase(1)
        keras.backend.manual_variable_initialization(True)

        model = ResNet18(10)
        model.build((None,32,32,3))
        model_ascent = ResNet18(10)
        model_ascent.build((None,32,32,3))
        model_descent = ResNet18(10)
        model_descent.build((None,32,32,3))

        targets = tf.placeholder(tf.float32, shape=[None, 10], name="y-input")
        
        predictions = model.output
        loss = tf.reduce_mean(
            keras.losses.categorical_crossentropy(targets, predictions))

        targets_ascent = tf.placeholder(tf.float32, shape=[None, 10], name="y-input_ascent")
        
        predictions_ascent = model_ascent.output
        loss_ascent = tf.reduce_mean(
            keras.losses.categorical_crossentropy(targets_ascent, predictions_ascent))

        targets_descent = tf.placeholder(tf.float32, shape=[None, 10], name="y-input_descent")
        
        predictions_descent = model_descent.output
        loss_descent = tf.reduce_mean(
            keras.losses.categorical_crossentropy(targets_descent, predictions_descent))

        learning_rate = tf.Variable(base_learning_rate, trainable=False)
        new_learning_rate = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
        update_learning_rate = tf.assign(learning_rate, new_learning_rate)

        optimizer = tf.train.SGD(learning_rate=learning_rate, momentum=0.9)

        optimizer_ascent = tf.train.SGD(learning_rate=ascent_descent_learning_rate, momentum=0.9, clipnorm=1.0)

        optimizer_descent = tf.train.SGD(learning_rate=ascent_descent_learning_rate, momentum=0.9, clipnorm=1.0)

        # Barrier to compute gradients after updating moving avg of batch norm
        with tf.control_dependencies(model.updates):
            barrier = tf.no_op(name="update_barrier")

        with tf.control_dependencies([barrier]):
            grads = optimizer.compute_gradients(
                loss,
                model.trainable_weights)
            grad_updates = optimizer.apply_gradients(grads)

        with tf.control_dependencies([grad_updates]):
            train_op = tf.identity(loss, name="train")

        with tf.control_dependencies(model_ascent.updates):
            barrier_ascent = tf.no_op(name="update_barrier_ascent")

        with tf.control_dependencies([barrier_ascent]):
            grads_ascent = optimizer_ascent.compute_gradients(
                loss_ascent,
                model_ascent.trainable_weights)
            grad_updates_ascent = optimizer_ascent.apply_gradients(grads_ascent)

        with tf.control_dependencies([grad_updates_ascent]):
            train_op_ascent = tf.identity(loss_ascent, name="train_ascent")

        with tf.control_dependencies(model_descent.updates):
            barrier_descent = tf.no_op(name="update_barrier_descent")

        with tf.control_dependencies([barrier_descent]):
            grads_descent = optimizer_descent.compute_gradients(
                loss_descent,
                model_descent.trainable_weights)
            grad_updates_descent = optimizer_descent.apply_gradients(grads_descent)

        with tf.control_dependencies([grad_updates_descent]):
            train_op_descent = tf.identity(loss_descent, name="train_descent")

        global_step = tf.get_variable('global_step', [],
                                      initializer=tf.constant_initializer(0),
                                      trainable=False)

        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
        train_dataset = train_dataset.batch(batch_size)

        correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(targets, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        learning_rate_multiplicator = tf.Variable(1.0, trainable=False)
        new_learning_rate_multiplicator = tf.placeholder(tf.float32, shape=[], name="new_learning_rate_multiplicator")
        update_learning_rate_multiplicator = tf.assign(learning_rate_multiplicator, new_learning_rate_multiplicator)
        
        if (FLAGS.task_index == 0):
            stochastic_sharpness_list = np.array([])

        init_op = tf.global_variables_initializer()

    sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                             global_step=global_step,
                             logdir="/tmp/train_logs",
                             save_model_secs=600,
                             init_op=init_op)

    print("Waiting for other servers")
    with sv.managed_session(server.target) as sess:
        keras.backend.set_session(sess)
        step_per_epoch = Y_train // batch_size
        max_step = step_per_epoch * epochs
        step_value = 0
        while not sv.should_stop() and step_value < max_step:
            import time
            start_time = time.time()
            (x_batch_train, y_batch_train) = train_dataset[step_value % step_per_epoch]
            # perform the operations we defined earlier on batch
            loss_value, step_value = sess.run(
                [train_op, global_step],
                feed_dict={
                    model.inputs[0]: x_batch_train,
                    targets: y_batch_train})

            if FLAGS.use_salr and FLAGS.task_index == 0:
                ###############
                ## FOR SALR ###
                ###############
                model_ascent.set_weights(model.get_weights())
                model_descent.set_weights(model.get_weights())

                for i in range(n_ascent):
                    loss_value_ascent = sess.run(train_op_ascent, feed_dict={model_ascent.inputs[0]: x_batch_train, targets_ascent: y_batch_train})

                    # grads_ascent = tf.gradients(loss_value_ascent, model_ascent.trainable_weights)
                    # clipped_grads_ascent, _ = tf.clip_by_global_norm(grads_ascent, 1.0)
                    # optimizer_ascent.apply_gradients(zip(clipped_grads_ascent, model_ascent.trainable_weights))
                
                for i in range(n_descent):
                    loss_value_descent = sess.run(train_op_descent, feed_dict={model_descent.inputs[0]: x_batch_train, targets_descent: y_batch_train})
                    
                    # grads_descent = tape_descent.gradient(loss_value_descent, model_descent.trainable_weights)
                    # clipped_grads_descent, _ = tf.clip_by_global_norm(grads_descent, 1.0)
                    # optimizer_descent.apply_gradients(zip(clipped_grads_descent, model_descent.trainable_weights))
                
                stochastic_sharpness = loss_value_ascent - loss_value_descent
                print("\n   Stochastic Sharpness %3.10f" % (stochastic_sharpness,))
                stochastic_sharpness_list =  np.append(stochastic_sharpness_list, stochastic_sharpness)
                median_sharpness = np.median(stochastic_sharpness_list)
                
                print("\n   Median Sharpness %3.10f" % (median_sharpness,))

                # sess.run(update_learning_rate_multiplicator, feed_dict={new_learning_rate_multiplicator: stochastic_sharpness / median_sharpness})

                # current_learning_rate_multiplicator = sess.run(learning_rate_multiplicator)
                sess.run(update_learning_rate, feed_dict={new_learning_rate: (stochastic_sharpness / median_sharpness * base_learning_rate)})
                # current_lr = K.get_value(optimizer.lr)
                # K.set_value(optimizer.lr, base_learning_rate * stochastic_sharpness / median_sharpness)
                # print("\n   Learning Rate %3.10f ==> %3.10f" % (current_lr, K.get_value(optimizer.lr),))


                ################
                ### END SALR ###
                ################

            if step_value % log_frequency == 0:
                elapsed_time = time.time() - start_time
                start_time = time.time()
                accuracy = sess.run(loss,
                                    feed_dict={
                                        model.inputs[0]: X_test,
                                        targets: Y_test})
                print("Step: %d," % (step_value + 1),
                    " Iteration: %2d," % step_value,
                    " Cost: %.4f," % loss_value,
                    " Accuracy: %.4f" % accuracy,
                    " AvgTime: %3.2fms" % float(elapsed_time * 1000 / log_frequency))
                

    sv.stop()
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


    print("train accuracy: %3.10f" % acc_train)
    print("test accuracy : %3.10f" % acc_test)
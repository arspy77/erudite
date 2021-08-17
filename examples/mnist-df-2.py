'''
Distributed Tensorflow 0.8.0 example of using data parallelism and share model parameters.
Trains a simple sigmoid neural network on mnist for 20 epochs on three machines using one parameter server. 

Change the hardcoded host urls below with your own hosts. 
Run like this: 

pc-01$ python example.py --job_name="ps" --task_index=0 
pc-02$ python example.py --job_name="worker" --task_index=0 
pc-03$ python example.py --job_name="worker" --task_index=1 
pc-04$ python example.py --job_name="worker" --task_index=2 

More details here: ischlag.github.io
'''

from __future__ import print_function

import argparse
import sys
import os
import ast
import time

import tensorflow as tf
import numpy as np
import sys
import time

class Empty:
  pass

FLAGS = Empty()

TF_CONFIG = ast.literal_eval(os.environ["TF_CONFIG"])
FLAGS.job_name = TF_CONFIG["task"]["type"]
FLAGS.task_index = TF_CONFIG["task"]["index"]
FLAGS.ps_hosts = ",".join(TF_CONFIG["cluster"]["ps"])
FLAGS.worker_hosts = ",".join(TF_CONFIG["cluster"]["worker"])
FLAGS.global_steps = 100
FLAGS.use_salr = (True if os.environ["use_salr"] == "True" else False) if "use_salr" in os.environ else True

ps_hosts = FLAGS.ps_hosts.split(",")
worker_hosts = FLAGS.worker_hosts.split(",")

cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

# start a server for a specific task
server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

# load mnist data set
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# config
batch_size = mnist.train.num_examples // 3
initial_learning_rate = 0.01 
training_epochs = 5
n_hidden = 10
logs_path = "/tmp/mnist/2"

if FLAGS.job_name == "ps":
    server.join()
elif FLAGS.job_name == "worker":

    # Between-graph replication
    with tf.device(
            tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % FLAGS.task_index, cluster=cluster)):

        # count the number of updates
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        # input images
        with tf.name_scope('input'):
            # None -> batch size can be any size, 784 -> flattened mnist image
            x = tf.placeholder(tf.float32, shape=[None, 784], name="x-input")
            # target 10 output classes
            y_ = tf.placeholder(tf.float32, shape=[None, 10], name="y-input")

        # model parameters will change during training so we use tf.Variable
        tf.set_random_seed(1)
        with tf.name_scope("weights"):
            W1 = tf.get_variable('W1',
                                 shape=(784, n_hidden),
                                 initializer=tf.contrib.layers.xavier_initializer())
            W2 = tf.get_variable('W2',
                                 shape=(n_hidden, 10),
                                 initializer=tf.contrib.layers.xavier_initializer())

        # bias
        with tf.name_scope("biases"):
            b1 = tf.Variable(tf.zeros([n_hidden]))
            b2 = tf.Variable(tf.zeros([10]))

        # implement model
        with tf.name_scope("softmax"):
            # y is our prediction
            z2 = tf.add(tf.matmul(x, W1), b1)
            a2 = tf.nn.sigmoid(z2)
            logits = tf.add(tf.matmul(a2, W2), b2)
            dropout_logits = tf.nn.dropout(logits, 0.3)

            softmax_logits = tf.nn.softmax(logits)

        # specify cost function
        with tf.name_scope('cross_entropy'):
            # this is our cost
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_)
            loss = tf.reduce_mean(cross_entropy)

        # specify optimizer
        with tf.name_scope('train'):
            # optimizer is an "operation" which we can execute in a session
            learning_rate = tf.Variable(initial_learning_rate, trainable=False)
            new_learning_rate = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
            update_learning_rate = tf.assign(learning_rate, new_learning_rate)
            grad_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            train_op = grad_op.minimize(loss, global_step=global_step)

        with tf.name_scope('Accuracy'):
            # accuracy
            correct_prediction = tf.equal(tf.argmax(softmax_logits, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # create a summary for our cost and accuracy
        tf.summary.scalar("cost", loss)
        tf.summary.scalar("accuracy", accuracy)

        # merge all summaries into a single "operation" which we can execute in a session
        summary_op = tf.summary.merge_all()

        ################################################################################################################
        # For SALR algorithm ###########################################################################################
        ################################################################################################################

        learning_rate_multiplicator = tf.Variable(1.0, trainable=False)
        new_learning_rate_multiplicator = tf.placeholder(tf.float32, shape=[], name="new_learning_rate_multiplicator")
        update_learning_rate_multiplicator = tf.assign(learning_rate_multiplicator, new_learning_rate_multiplicator)
        
        base_learning_rate = 0.002
        n_ascent = 5
        n_descent = 5


        x_ascent = tf.placeholder(tf.float32, shape=[None, 784], name="x-input_ascent")
        y__ascent = tf.placeholder(tf.float32, shape=[None, 10], name="y-input_ascent")

        W1_ascent = tf.get_variable('W1_ascent',
                                 shape=(784, n_hidden),
                                 initializer=tf.contrib.layers.xavier_initializer())
        W2_ascent = tf.get_variable('W2_ascent',
                                 shape=(n_hidden, 10),
                                 initializer=tf.contrib.layers.xavier_initializer())
                                
        b1_ascent = tf.Variable(tf.zeros([n_hidden]))
        b2_ascent = tf.Variable(tf.zeros([10]))

        assign_W1_ascent = W1_ascent.assign(W1)
        assign_W2_ascent = W2_ascent.assign(W2)
        assign_b1_ascent = b1_ascent.assign(b1)
        assign_b2_ascent = b2_ascent.assign(b2)

        z2_ascent = tf.add(tf.matmul(x_ascent, W1_ascent), b1_ascent)
        a2_ascent = tf.nn.sigmoid(z2_ascent)
        logits_ascent = tf.add(tf.matmul(a2_ascent, W2_ascent), b2_ascent)
        dropout_logits_ascent = tf.nn.dropout(logits_ascent, 0.3)

        softmax_logits_ascent = tf.nn.softmax(logits_ascent)

        cross_entropy_ascent = tf.nn.softmax_cross_entropy_with_logits(logits=logits_ascent, labels=y__ascent)
        loss_ascent = tf.reduce_mean(cross_entropy_ascent)
        ascent_loss_op = tf.reduce_sum(loss_ascent) ###

        opt_ascent = tf.train.GradientDescentOptimizer(learning_rate=-base_learning_rate)
        grads_and_vars_ascent = tf.gradients(cross_entropy_ascent, [W1_ascent, W2_ascent, b1_ascent, b2_ascent])
        capped_grads_and_vars_ascent, _ = tf.clip_by_global_norm(grads_and_vars_ascent, 1) 
        train_op_ascent = opt_ascent.apply_gradients(zip(capped_grads_and_vars_ascent, [W1_ascent, W2_ascent, b1_ascent, b2_ascent]))



        x_descent = tf.placeholder(tf.float32, shape=[None, 784], name="x-input_descent")
        y__descent = tf.placeholder(tf.float32, shape=[None, 10], name="y-input_descent")

        W1_descent = tf.get_variable('W1_descent',
                                 shape=(784, n_hidden),
                                 initializer=tf.contrib.layers.xavier_initializer())
        W2_descent = tf.get_variable('W2_descent',
                                 shape=(n_hidden, 10),
                                 initializer=tf.contrib.layers.xavier_initializer())
                                
        b1_descent = tf.Variable(tf.zeros([n_hidden]))
        b2_descent = tf.Variable(tf.zeros([10]))

        assign_W1_descent = W1_descent.assign(W1)
        assign_W2_descent = W2_descent.assign(W2)
        assign_b1_descent = b1_descent.assign(b1)
        assign_b2_descent = b2_descent.assign(b2)

        z2_descent = tf.add(tf.matmul(x_descent, W1_descent), b1_descent)
        a2_descent = tf.nn.sigmoid(z2_descent)
        logits_descent = tf.add(tf.matmul(a2_descent, W2_descent), b2_descent)
        dropout_logits_descent = tf.nn.dropout(logits_descent, 0.3)

        softmax_logits_descent = tf.nn.softmax(logits_descent)

        cross_entropy_descent = tf.nn.softmax_cross_entropy_with_logits(logits=logits_descent, labels=y__descent)
        loss_descent = tf.reduce_mean(cross_entropy_descent)
        descent_loss_op = tf.reduce_sum(loss_descent) 

        opt_descent = tf.train.GradientDescentOptimizer(learning_rate=base_learning_rate)
        grads_and_vars_descent = tf.gradients(cross_entropy_descent, [W1_descent, W2_descent, b1_descent, b2_descent])
        capped_grads_and_vars_descent, _ = tf.clip_by_global_norm(grads_and_vars_descent, 1) 
        train_op_descent = opt_descent.apply_gradients(zip(capped_grads_and_vars_descent, [W1_descent, W2_descent, b1_descent, b2_descent]))

        init_op = tf.initialize_all_variables()
        print("Variables initialized ...")

    sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0), global_step=global_step, init_op=init_op)

    begin_time = time.time()
    frequency = 100
    if (FLAGS.task_index == 0):
      stochastic_sharpness_list = np.array([])
    with sv.prepare_or_wait_for_session(server.target) as sess:
        # create log writer object (this will log on every machine)
        writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

        # perform training cycles
        start_time = time.time()
        for epoch in range(training_epochs):

            # number of batches in one epoch
            batch_count = int(mnist.train.num_examples / batch_size)


            count = 0
            for i in range(batch_count):
                batch_x, batch_y = mnist.train.next_batch(batch_size)

                # perform the operations we defined earlier on batch
                _, cost, summary, step, train_accuracy = sess.run([train_op, loss, summary_op, global_step, accuracy],
                    feed_dict={x: batch_x, y_: batch_y})
                writer.add_summary(summary, step)

                count += 1
                if count % frequency == 0 or i + 1 == batch_count:
                    elapsed_time = time.time() - start_time
                    start_time = time.time()
                    print("Step: %d," % (step + 1), " Epoch: %2d," % (epoch + 1),
                          " Batch: %3d of %3d," % (i + 1, batch_count), " Cost: %.4f," % cost,
                          " Train acc %2.2f" % (train_accuracy * 100),
                          " AvgTime: %3.2fms" % float(elapsed_time * 1000 / frequency))
                    count = 0
                
                # Updates learning rate with SALR algorithm
                if FLAGS.use_salr and FLAGS.task_index == 0 and count % 2 == 0:
                    sess.run(assign_W1_ascent)
                    sess.run(assign_b1_ascent)
                    sess.run(assign_W1_descent)
                    sess.run(assign_b1_descent)
                    sess.run(assign_W2_ascent)
                    sess.run(assign_b2_ascent)
                    sess.run(assign_W2_descent)
                    sess.run(assign_b2_descent)
                    
                    for i in range(n_ascent):
                        sess.run(train_op_ascent, feed_dict={x_ascent: batch_x, y__ascent: batch_y})
                    for i in range(n_descent):
                        sess.run(train_op_descent, feed_dict={x_descent: batch_x, y__descent: batch_y})
                    
                    descent_loss = 0
                    ascent_loss = 0
                    for i in range(batch_size):
                        descent_loss += sess.run(cross_entropy_descent, feed_dict={x_descent: [batch_x[i]], y__descent: [batch_y[i]]})
                        ascent_loss += sess.run(cross_entropy_ascent, feed_dict={x_ascent: [batch_x[i]], y__ascent: [batch_y[i]]}) 
                    # descent_loss = sess.run(descent_loss_op, feed_dict={x_descent: batch_x, y__descent: batch_y})
                    # ascent_loss = sess.run(ascent_loss_op, feed_dict={x_ascent: batch_x, y__ascent: batch_y})
                        
                    stochastic_sharpness = float(ascent_loss - descent_loss) / batch_size
                    print("asc loss : %3.10f" % ascent_loss)
                    print("desc loss : %3.10f" % descent_loss)
                    print(stochastic_sharpness)
                    print("^ss msv")
                    stochastic_sharpness_list =  np.append(stochastic_sharpness_list, stochastic_sharpness)

                    median_sharpness = np.median(stochastic_sharpness_list)
                    print(median_sharpness)
                    sess.run(update_learning_rate_multiplicator, feed_dict={new_learning_rate_multiplicator: stochastic_sharpness / median_sharpness})
                
                current_learning_rate_multiplicator = sess.run(learning_rate_multiplicator)
                sess.run(update_learning_rate, feed_dict={new_learning_rate: (current_learning_rate_multiplicator * initial_learning_rate)})

        print("Test-Accuracy: %2.2f" % (sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}) *100))
        print("Total Time: %3.2fs" % float(time.time() - begin_time))
        print("Final Cost: %.4f" % cost)

    sv.stop()
    print("done")
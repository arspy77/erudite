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

FLAGS.use_salr = (True if os.environ["use_salr"] == "True" else False) if "use_salr" in os.environ else True
FLAGS.epoch = int(os.environ["epoch"]) if "epoch" in os.environ else 100

ps_hosts = FLAGS.ps_hosts.split(",")
worker_hosts = FLAGS.worker_hosts.split(",")

cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

# start a server for a specific task
server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

# load mnist data set
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# config
batch_size = 100
initial_learning_rate = 0.01 
training_epochs = FLAGS.epoch
n_hidden_1 = 200
n_hidden_2 = 80
# n_hidden_3 = 1000
# n_hidden_4 = 1000
# n_hidden_5 = 1000
logs_path = "/tmp/mnist/2"

if FLAGS.job_name == "ps":
    server.join()
elif FLAGS.job_name == "worker":

    # Between-graph replication
    with tf.device(
            tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % FLAGS.task_index, cluster=cluster)):

        # count the number of updates
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        with tf.name_scope('input'):
            # None -> batch size can be any size, 784 -> flattened mnist image
            x = tf.placeholder(tf.float32, shape=[None, 784], name="x-input")
            # target 10 output classes
            y_ = tf.placeholder(tf.float32, shape=[None, 10], name="y-input")

        # model parameters will change during training so we use tf.Variable
        tf.set_random_seed(1)
        with tf.name_scope("weights"):
            W1 = tf.get_variable('W1',
                                shape=(784, n_hidden_1),
                                initializer=tf.contrib.layers.xavier_initializer())
            W2 = tf.get_variable('W2',
                                shape=(n_hidden_1, n_hidden_2),
                                initializer=tf.contrib.layers.xavier_initializer())
            W3 = tf.get_variable('W3',
                                shape=(n_hidden_2, 10),
                                initializer=tf.contrib.layers.xavier_initializer())
            # W4 = tf.get_variable('W4',
            #                      shape=(n_hidden_3, n_hidden_4),
            #                      initializer=tf.contrib.layers.xavier_initializer())
            # W5 = tf.get_variable('W5',
            #                      shape=(n_hidden_4, n_hidden_5),
            #                      initializer=tf.contrib.layers.xavier_initializer())
            # W6 = tf.get_variable('W6',
            #                      shape=(n_hidden_5, 10),
            #                      initializer=tf.contrib.layers.xavier_initializer())

        # bias
        with tf.name_scope("biases"):
            b1 = tf.Variable(tf.zeros([n_hidden_1]))
            b2 = tf.Variable(tf.zeros([n_hidden_2]))
            b3 = tf.Variable(tf.zeros([10]))
            # b4 = tf.Variable(tf.zeros([n_hidden_4]))
            # b5 = tf.Variable(tf.zeros([n_hidden_5]))
            # b6 = tf.Variable(tf.zeros([10]))

        # implement model
        with tf.name_scope("softmax"):
            # y is our prediction
            z2 = tf.add(tf.matmul(x, W1), b1)
            a2 = tf.nn.sigmoid(z2)
            z3 = tf.add(tf.matmul(a2, W2), b2)
            a3 = tf.nn.sigmoid(z3)
            # z4 = tf.add(tf.matmul(a3, W3), b3)
            # a4 = tf.nn.sigmoid(z4)
            # z5 = tf.add(tf.matmul(a4, W4), b4)
            # a5 = tf.nn.sigmoid(z5)
            # z6 = tf.add(tf.matmul(a5, W5), b5)
            # a6 = tf.nn.sigmoid(z6)
            logits = tf.add(tf.matmul(a3, W3), b3)
            dropout_logits = tf.nn.dropout(logits, 0.5)

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

        # # create a summary for our cost and accuracy
        # tf.summary.scalar("cost", loss)
        # tf.summary.scalar("accuracy", accuracy)

        # # merge all summaries into a single "operation" which we can execute in a session
        # summary_op = tf.summary.merge_all()

        ################################################################################################################
        # For SALR algorithm ###########################################################################################
        ################################################################################################################

        # learning_rate_multiplicator = tf.Variable(1.0, trainable=False)
        # new_learning_rate_multiplicator = tf.placeholder(tf.float32, shape=[], name="new_learning_rate_multiplicator")
        # update_learning_rate_multiplicator = tf.assign(learning_rate_multiplicator, new_learning_rate_multiplicator)
        
        
        # median_sharpness_list = []
        # new_median_sharpness_list = []
        # update_median_sharpness_list = []
        # for i in range(len(worker_hosts)):
        #     median_sharpness_list.append(tf.Variable(0., trainable=False))
        #     new_median_sharpness_list.append(tf.placeholder(tf.float32, shape=[], name="new_median_sharpness_" + str(i)))
        #     update_median_sharpness_list.append(tf.assign(median_sharpness_list[i], new_median_sharpness_list[i]))
        
        mean_sharpness = tf.Variable(0., trainable=False)
        new_mean_sharpness = tf.placeholder(tf.float32, shape=[], name="new_mean_sharpness")
        update_mean_sharpness = tf.assign(mean_sharpness, new_mean_sharpness)

        sharpness_count = tf.Variable(0, trainable=False)
        new_sharpness_count = tf.placeholder(tf.int32, shape=[], name="new_sharpness_count")
        update_sharpness_count = tf.assign(sharpness_count, new_sharpness_count)

        base_learning_rate = 0.002
        n_ascent = 5
        n_descent = 5


        x_ascent = tf.placeholder(tf.float32, shape=[None, 784], name="x-input_ascent")
        y__ascent = tf.placeholder(tf.float32, shape=[None, 10], name="y-input_ascent")

        W1_ascent = tf.get_variable('W1_ascent',
                                shape=(784, n_hidden_1),
                                initializer=tf.contrib.layers.xavier_initializer())
        W2_ascent = tf.get_variable('W2_ascent',
                                shape=(n_hidden_1, n_hidden_2),
                                initializer=tf.contrib.layers.xavier_initializer())
        W3_ascent = tf.get_variable('W3_ascent',
                                shape=(n_hidden_2, 10),
                                initializer=tf.contrib.layers.xavier_initializer())
        # W4_ascent = tf.get_variable('W4_ascent',
        #                          shape=(n_hidden_3, n_hidden_4),
        #                          initializer=tf.contrib.layers.xavier_initializer())
        # W5_ascent = tf.get_variable('W5_ascent',
        #                          shape=(n_hidden_4, n_hidden_5),
        #                          initializer=tf.contrib.layers.xavier_initializer())
        # W6_ascent = tf.get_variable('W6_ascent',
        #                          shape=(n_hidden_5, 10),
        #                          initializer=tf.contrib.layers.xavier_initializer())
                                
        b1_ascent = tf.Variable(tf.zeros([n_hidden_1]))
        b2_ascent = tf.Variable(tf.zeros([n_hidden_2]))
        b3_ascent = tf.Variable(tf.zeros([10]))
        # b4_ascent = tf.Variable(tf.zeros([n_hidden_4]))
        # b5_ascent = tf.Variable(tf.zeros([n_hidden_5]))
        # b6_ascent = tf.Variable(tf.zeros([10]))

        assign_W1_ascent = W1_ascent.assign(W1)
        assign_W2_ascent = W2_ascent.assign(W2)
        assign_W3_ascent = W3_ascent.assign(W3)
        # assign_W4_ascent = W4_ascent.assign(W4)
        # assign_W5_ascent = W5_ascent.assign(W5)
        # assign_W6_ascent = W6_ascent.assign(W6)
        assign_b1_ascent = b1_ascent.assign(b1)
        assign_b2_ascent = b2_ascent.assign(b2)
        assign_b3_ascent = b3_ascent.assign(b3)
        # assign_b4_ascent = b4_ascent.assign(b4)
        # assign_b5_ascent = b5_ascent.assign(b5)
        # assign_b6_ascent = b6_ascent.assign(b6)

        z2_ascent = tf.add(tf.matmul(x_ascent, W1_ascent), b1_ascent)
        a2_ascent = tf.nn.sigmoid(z2_ascent)
        z3_ascent = tf.add(tf.matmul(a2_ascent, W2_ascent), b2_ascent)
        a3_ascent = tf.nn.sigmoid(z3_ascent)
        # z4_ascent = tf.add(tf.matmul(a3_ascent, W3_ascent), b3_ascent)
        # a4_ascent = tf.nn.sigmoid(z4_ascent)
        # z5_ascent = tf.add(tf.matmul(a4_ascent, W4_ascent), b4_ascent)
        # a5_ascent = tf.nn.sigmoid(z5_ascent)
        # z6_ascent = tf.add(tf.matmul(a5_ascent, W5_ascent), b5_ascent)
        # a6_ascent = tf.nn.sigmoid(z6_ascent)
        logits_ascent = tf.add(tf.matmul(a3_ascent, W3_ascent), b3_ascent)
        dropout_logits_ascent = tf.nn.dropout(logits_ascent, 0.5)

        softmax_logits_ascent = tf.nn.softmax(logits_ascent)

        cross_entropy_ascent = tf.nn.softmax_cross_entropy_with_logits(logits=logits_ascent, labels=y__ascent)
        loss_ascent = tf.reduce_mean(cross_entropy_ascent)

        opt_ascent = tf.train.GradientDescentOptimizer(learning_rate=-base_learning_rate)
        grads_and_vars_ascent = tf.gradients(loss_ascent, [W1_ascent, W2_ascent, W3_ascent, b1_ascent, b2_ascent, b3_ascent])
        capped_grads_and_vars_ascent, _ = tf.clip_by_global_norm(grads_and_vars_ascent, 1)
        train_op_ascent = opt_ascent.apply_gradients(zip(capped_grads_and_vars_ascent, [W1_ascent, W2_ascent, W3_ascent, b1_ascent, b2_ascent, b3_ascent]))



        x_descent = tf.placeholder(tf.float32, shape=[None, 784], name="x-input_descent")
        y__descent = tf.placeholder(tf.float32, shape=[None, 10], name="y-input_descent")

        W1_descent = tf.get_variable('W1_descent',
                                shape=(784, n_hidden_1),
                                initializer=tf.contrib.layers.xavier_initializer())
        W2_descent = tf.get_variable('W2_descent',
                                shape=(n_hidden_1, n_hidden_2),
                                initializer=tf.contrib.layers.xavier_initializer())
        W3_descent = tf.get_variable('W3_descent',
                                shape=(n_hidden_2, 10),
                                initializer=tf.contrib.layers.xavier_initializer())
        # W4_descent = tf.get_variable('W4_descent',
        #                          shape=(n_hidden_3, n_hidden_4),
        #                          initializer=tf.contrib.layers.xavier_initializer())
        # W5_descent = tf.get_variable('W5_descent',
        #                          shape=(n_hidden_4, n_hidden_5),
        #                          initializer=tf.contrib.layers.xavier_initializer())
        # W6_descent = tf.get_variable('W6_descent',
        #                          shape=(n_hidden_5, 10),
        #                          initializer=tf.contrib.layers.xavier_initializer())
                                
        b1_descent = tf.Variable(tf.zeros([n_hidden_1]))
        b2_descent = tf.Variable(tf.zeros([n_hidden_2]))
        b3_descent = tf.Variable(tf.zeros([10]))
        # b4_descent = tf.Variable(tf.zeros([n_hidden_4]))
        # b5_descent = tf.Variable(tf.zeros([n_hidden_5]))
        # b6_descent = tf.Variable(tf.zeros([10]))

        assign_W1_descent = W1_descent.assign(W1)
        assign_W2_descent = W2_descent.assign(W2)
        assign_W3_descent = W3_descent.assign(W3)
        # assign_W4_descent = W4_descent.assign(W4)
        # assign_W5_descent = W5_descent.assign(W5)
        # assign_W6_descent = W6_descent.assign(W6)
        assign_b1_descent = b1_descent.assign(b1)
        assign_b2_descent = b2_descent.assign(b2)
        assign_b3_descent = b3_descent.assign(b3)
        # assign_b4_descent = b4_descent.assign(b4)
        # assign_b5_descent = b5_descent.assign(b5)
        # assign_b6_descent = b6_descent.assign(b6)

        z2_descent = tf.add(tf.matmul(x_descent, W1_descent), b1_descent)
        a2_descent = tf.nn.sigmoid(z2_descent)
        z3_descent = tf.add(tf.matmul(a2_descent, W2_descent), b2_descent)
        a3_descent = tf.nn.sigmoid(z3_descent)
        # z4_descent = tf.add(tf.matmul(a3_descent, W3_descent), b3_descent)
        # a4_descent = tf.nn.sigmoid(z4_descent)
        # z5_descent = tf.add(tf.matmul(a4_descent, W4_descent), b4_descent)
        # a5_descent = tf.nn.sigmoid(z5_descent)
        # z6_descent = tf.add(tf.matmul(a5_descent, W5_descent), b5_descent)
        # a6_descent = tf.nn.sigmoid(z6_descent)
        logits_descent = tf.add(tf.matmul(a3_descent, W3_descent), b3_descent)
        dropout_logits_descent = tf.nn.dropout(logits_descent, 0.5)

        softmax_logits_descent = tf.nn.softmax(logits_descent)

        cross_entropy_descent = tf.nn.softmax_cross_entropy_with_logits(logits=logits_descent, labels=y__descent)
        loss_descent = tf.reduce_mean(cross_entropy_descent)

        opt_descent = tf.train.GradientDescentOptimizer(learning_rate=base_learning_rate)
        grads_and_vars_descent = tf.gradients(loss_descent, [W1_descent, W2_descent, W3_descent, b1_descent, b2_descent, b3_descent])
        capped_grads_and_vars_descent, _ = tf.clip_by_global_norm(grads_and_vars_descent, 1)
        train_op_descent = opt_descent.apply_gradients(zip(capped_grads_and_vars_descent, [W1_descent, W2_descent, W3_descent, b1_descent, b2_descent, b3_descent]))

        init_op = tf.initialize_all_variables()
        print("Variables initialized ...")

    # sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0), global_step=global_step, init_op=init_op,
    #                             config=tf.ConfigProto(
    #                                            device_filters=["/job:ps", "/job:worker/task:%d" % FLAGS.task_index]
    #                             )
    #                         )
    
    begin_time = time.time()
    frequency = 100
    # stochastic_sharpness_list = np.array([])

    with tf.train.MonitoredTrainingSession(master=server.target,
                                           is_chief=(FLAGS.task_index == 0),
                                           config=tf.ConfigProto(
                                               device_filters=["/job:ps", "/job:worker/task:%d" % FLAGS.task_index]
                                                )
                                           ) as sess:
        # create log writer object (this will log on every machine)
        # writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

        # perform training cycles
        start_time = time.time()
        step = 0
        batch_count = int(mnist.train.num_examples / batch_size)
        total_step = training_epochs*batch_count
        while step < total_step:

            count = 0
            for i in range(batch_count):
                batch_x, batch_y = mnist.train.next_batch(batch_size)

                # perform the operations we defined earlier on batch
                _, cost, step, train_accuracy = sess.run([train_op, loss, global_step, accuracy],
                    feed_dict={x: batch_x, y_: batch_y})
                # writer.add_summary(summary, step)

                count += 1
                if count % frequency == 0 or i + 1 == batch_count:
                    elapsed_time = time.time() - start_time
                    start_time = time.time()
                    print("Step: %d," % (step + 1), " Epoch: %2d," % (step//batch_count + 1),
                        #" Batch: %3d of %3d," % (i + 1, batch_count), " Cost: %.4f," % cost,
                        " Train acc %2.2f" % (train_accuracy * 100),
                        " AvgTime: %3.2fms" % float(elapsed_time * 1000 / frequency),
                        " Learning Rate: %3.10f " % sess.run(learning_rate))
                    count = 0
                
                # Updates learning rate with SALR algorithm
                if FLAGS.use_salr and count % 2 == 0:
                    sess.run(assign_W1_ascent)
                    sess.run(assign_b1_ascent)
                    sess.run(assign_W1_descent)
                    sess.run(assign_b1_descent)
                    sess.run(assign_W2_ascent)
                    sess.run(assign_b2_ascent)
                    sess.run(assign_W2_descent)
                    sess.run(assign_b2_descent)
                    sess.run(assign_W3_ascent)
                    sess.run(assign_b3_ascent)
                    sess.run(assign_W3_descent)
                    sess.run(assign_b3_descent)
                    # sess.run(assign_W4_ascent)
                    # sess.run(assign_b4_ascent)
                    # sess.run(assign_W4_descent)
                    # sess.run(assign_b4_descent)
                    # sess.run(assign_W5_ascent)
                    # sess.run(assign_b5_ascent)
                    # sess.run(assign_W5_descent)
                    # sess.run(assign_b5_descent)
                    
                    for i in range(n_ascent):
                        sess.run(train_op_ascent, feed_dict={x_ascent: batch_x, y__ascent: batch_y})
                    for i in range(n_descent):
                        sess.run(train_op_descent, feed_dict={x_descent: batch_x, y__descent: batch_y})

                    ascent_loss = sess.run(loss_ascent, feed_dict={x_ascent: batch_x, y__ascent: batch_y})
                    descent_loss = sess.run(loss_descent, feed_dict={x_descent: batch_x, y__descent: batch_y})
                        
                    stochastic_sharpness = float(ascent_loss - descent_loss) / batch_size
                    
                    current_mean_sharpness = sess.run(mean_sharpness)
                    current_sharpness_count = sess.run(sharpness_count)

                    
                    current_mean_sharpness = (current_mean_sharpness*current_sharpness_count + stochastic_sharpness) / (current_sharpness_count + 1)
                    current_sharpness_count = current_sharpness_count + 1

                    sess.run(update_mean_sharpness, feed_dict={new_mean_sharpness: current_mean_sharpness})
                    sess.run(update_sharpness_count, feed_dict={new_sharpness_count: current_sharpness_count})

                    if current_mean_sharpness != 0:
                        sess.run(update_learning_rate, feed_dict={new_learning_rate: stochastic_sharpness / current_mean_sharpness * initial_learning_rate})
    
        if (FLAGS.task_index == 0):
            print("Test-Accuracy: %2.10f" % (sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}) *100))
            print("Total Time: %3.10fs" % float(time.time() - begin_time))
    
import argparse
import sys
import os
import ast
import time

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf
import numpy as np

class Empty:
  pass

batch_size = 10000

FLAGS = Empty()

def main(_):
  ps_hosts = FLAGS.ps_hosts.split(",")
  worker_hosts = FLAGS.worker_hosts.split(",")
  
  # Create a cluster from the parameter server and worker hosts.
  cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

  # Create and start a server for the local task.
  server = tf.train.Server(cluster,
                           job_name=FLAGS.job_name,
                           task_index=FLAGS.task_index)

  if FLAGS.job_name == "ps":
    server.join()
  elif FLAGS.job_name == "worker":

    # Assigns ops to the local worker by default.
    with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % FLAGS.task_index,
        cluster=cluster)):

      # Build model...
      x = tf.placeholder(tf.float32, [None, 784])
      W = tf.Variable(tf.zeros([784, 10]))
      b = tf.Variable(tf.zeros([10]))
      y = tf.nn.softmax(tf.matmul(x, W) + b)
      y_ = tf.placeholder(tf.float32, [None, 10])
      cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
      
      learning_rate = tf.Variable(0.05, trainable=False)
      new_learning_rate = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
      update_learning_rate = tf.assign(learning_rate, new_learning_rate)

      global_step = tf.train.get_or_create_global_step()
      train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy, global_step=global_step)

      # For SALR algorithm
      stochastic_sharpness_list = tf.Variable([-9999.9,9999.0])
      new_stochastic_sharpness = tf.placeholder(tf.float32, shape=[], name="new_stochastic_sharpness")
      concat_to_stochastic_sharpness_list = tf.concat([stochastic_sharpness_list, [new_stochastic_sharpness]], 0)
      get_stochastic_sharpness_median_op = tf.contrib.distributions.percentile(stochastic_sharpness_list,50.)
      
      base_learning_rate = 0.05
      n_ascent = 1
      n_descent = 1
      freq = 20

      x_ascent = tf.placeholder(tf.float32, [None, 784])
      W_ascent = tf.Variable(tf.zeros([784, 10]))
      assign_W_ascent = W_ascent.assign(W)
      b_ascent = tf.Variable(tf.zeros([10]))
      assign_b_ascent = b_ascent.assign(b)
      y_ascent = tf.nn.softmax(tf.matmul(x_ascent, W_ascent) + b_ascent)
      y__ascent = tf.placeholder(tf.float32, [None, 10])
      cross_entropy_ascent = tf.reduce_mean(-tf.reduce_sum(y__ascent * tf.log(y_ascent), reduction_indices=[1]))
      
      
      # gradients_ascent = tf.gradients(cross_entropy_ascent, [W_ascent, b_ascent])
      # grads_ascent = [tf.clip_by_norm(grad, 1)  for grad in gradients_ascent]
      # train_op_ascent = tf.train.GradientDescentOptimizer(learning_rate=base_learning_rate).apply_gradients(grads_ascent)

      opt_ascent = tf.train.GradientDescentOptimizer(learning_rate=base_learning_rate)
      grads_and_vars_ascent = opt_ascent.compute_gradients(cross_entropy_ascent, [W_ascent, b_ascent])
      capped_grads_and_vars_ascent = [(tf.clip_by_norm(gv[0], 1), gv[1]) for gv in grads_and_vars_ascent]
      train_op_ascent = opt_ascent.apply_gradients(capped_grads_and_vars_ascent)

      x_descent = tf.placeholder(tf.float32, [None, 784])
      W_descent = tf.Variable(tf.zeros([784, 10]))
      assign_W_descent = W_descent.assign(W)
      b_descent = tf.Variable(tf.zeros([10]))
      assign_b_descent = b_descent.assign(b)
      y_descent = tf.nn.softmax(tf.matmul(x_descent, W_descent) + b_descent)
      y__descent = tf.placeholder(tf.float32, [None, 10])
      cross_entropy_descent = tf.reduce_mean(-tf.reduce_sum(y__descent * tf.log(y_descent), reduction_indices=[1]))
      
      # gradients_descent = tf.gradients(cross_entropy_descent, [W_descent, b_descent])
      # grads_descent = [tf.clip_by_norm(grad, 1)  for grad in gradients_descent]
      # train_op_descent = tf.train.GradientDescentOptimizer(learning_rate=base_learning_rate).apply_gradients(grads_descent)

      opt_descent = tf.train.GradientDescentOptimizer(learning_rate=base_learning_rate)
      grads_and_vars_descent = opt_descent.compute_gradients(cross_entropy_descent, [W_descent, b_descent])
      capped_grads_and_vars_descent = [(tf.clip_by_norm(gv[0], 1), gv[1]) for gv in grads_and_vars_descent]
      train_op_descent = opt_descent.apply_gradients(capped_grads_and_vars_descent)

      # For Test Accuracy Checking
      correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # The StopAtStepHook handles stopping after running given steps.
    hooks=[tf.train.StopAtStepHook(last_step=FLAGS.global_steps)]

    # The MonitoredTrainingSession takes care of session initialization,
    # restoring from a checkpoint, saving to a checkpoint, and closing when done
    # or an error occurs.
    begin_time = time.time()
    test_accuracy = 0
    with tf.train.MonitoredTrainingSession(master=server.target,
                                           is_chief=(FLAGS.task_index == 0),
                                           config=tf.ConfigProto(
                                               device_filters=["/job:ps", "/job:worker/task:%d" % FLAGS.task_index]
                                           ),
                                           hooks=hooks) as mon_sess:

      while not mon_sess.should_stop():
        print("A")
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        _, step = mon_sess.run([train_step, global_step], feed_dict={x: batch_xs, y_: batch_ys})
        
        # Updates learning rate with SALR algorithm
        print("B")
        if step % freq == freq-1 and FLAGS.use_salr:
          print("C")
          if not mon_sess.should_stop():
            mon_sess.run(assign_W_ascent)
          if not mon_sess.should_stop():
            mon_sess.run(assign_b_ascent)
          if not mon_sess.should_stop():
            mon_sess.run(assign_W_descent)
          if not mon_sess.should_stop():
            mon_sess.run(assign_b_descent)
            print("D")
          for i in range(n_ascent):
            if not mon_sess.should_stop():
              mon_sess.run(train_op_ascent, feed_dict={x_ascent: batch_xs, y__ascent: batch_ys})
            print("E")
          for i in range(n_descent):
            if not mon_sess.should_stop():
              mon_sess.run(train_op_descent, feed_dict={x_descent: batch_xs, y__descent: batch_ys})
            print("F")
          descent_loss = 0
          ascent_loss = 0
          for i in range(batch_size):
            if not mon_sess.should_stop():
              descent_loss += mon_sess.run(cross_entropy_descent, feed_dict={x_descent: [batch_xs[i]], y__descent: [batch_ys[i]]})
              print("G")
            if not mon_sess.should_stop():
              ascent_loss += mon_sess.run(cross_entropy_ascent, feed_dict={x_ascent: [batch_xs[i]], y__ascent: [batch_ys[i]]}) 
              print("H")
          stochastic_sharpness = float(ascent_loss - descent_loss) / batch_size
          if not mon_sess.should_stop():
            mon_sess.run(concat_to_stochastic_sharpness_list, feed_dict={new_stochastic_sharpness: stochastic_sharpness})
            print("I")
          median_sharpness = 0
          if not mon_sess.should_stop():
            median_sharpness = mon_sess.run(get_stochastic_sharpness_median_op)
            print("J")
            #median_sharpness = mon_sess.run(median_sharpness_op)
          current_learning_rate = 0
          if not mon_sess.should_stop():
            current_learning_rate = mon_sess.run(learning_rate)
            print("K")
          if not mon_sess.should_stop():
            current_learning_rate = mon_sess.run(update_learning_rate, feed_dict={new_learning_rate: (stochastic_sharpness / median_sharpness * current_learning_rate)})
            print("L")
        
        if step > 55 and not mon_sess.should_stop():
          test_accuracy = mon_sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
        if not mon_sess.should_stop():
          print("M")
          print(mon_sess.run(learning_rate))
          
           

        # for i, row in enumerate(temp_W):
        #   for j, element in enumerate(row):
        #     W_ascent[i, j] = temp_W[i, j] 


        #sys.stderr.write('global_step: '+str(step))
        #sys.stderr.write('\n')
      print("Total Time: %3.10fs" % float(time.time() - begin_time))
      print("Test-Accuracy: %2.10f" % test_accuracy)


if __name__ == "__main__":
  TF_CONFIG = ast.literal_eval(os.environ["TF_CONFIG"])
  FLAGS.job_name = TF_CONFIG["task"]["type"]
  FLAGS.task_index = TF_CONFIG["task"]["index"]
  FLAGS.ps_hosts = ",".join(TF_CONFIG["cluster"]["ps"])
  FLAGS.worker_hosts = ",".join(TF_CONFIG["cluster"]["worker"])
  FLAGS.global_steps = 10000
  FLAGS.use_salr = (True if os.environ["use_salr"] == "True" else False) if "use_salr" in os.environ else True
  #FLAGS.global_steps = int(os.environ["global_steps"]) if "global_steps" in os.environ else 100000
  tf.app.run(main=main, argv=[sys.argv[0]])
  
#!/usr/bin/env python
# coding=utf-8
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("dataset/mnist",one_hot=True)
sess = tf.InteractiveSession()
x = tf.placeholder(dtype=tf.float32,shape=[None,784])
y_ = tf.placeholder(dtype=tf.float32,shape=[None,10])
image_reshaped_input = tf.reshape(x,[-1,28,28,1])
tf.summary.image('input',image_reshaped_input,10)

W1 = tf.Variable(tf.truncated_normal([784,300],stddev=0.1))
b1 = tf.Variable(tf.zeros([300]))
hidden1 = tf.nn.relu(tf.matmul(x,W1)+b1)
keep_prob = tf.placeholder(tf.float32)
hidden1_drop = tf.nn.dropout(hidden1,keep_prob)
W2 = tf.Variable(tf.zeros([300,10]))
b2 = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(hidden1_drop,W2)+b2)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))
tf.summary.scalar("cross_entropy",cross_entropy)

train_step = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)
prediction_result = tf.argmax(y,1)
tf.summary.histogram("prediction_result",prediction_result)

correct_prediction = tf.equal(tf.argmax(y_,1),prediction_result)
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
tf.summary.scalar("accuracy",accuracy)

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("./log/mnist",sess.graph)

tf.global_variables_initializer().run()
for i in range(3000):
        batch_xs,batch_ys = mnist.train.next_batch(100) 
        _,summary = sess.run((train_step,merged),feed_dict={x:batch_xs,y_:batch_ys,keep_prob:0.75})
        writer.add_summary(summary,i)
print("The final resultis"+str(accuracy.eval({x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0})))

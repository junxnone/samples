#!/usr/bin/env python
# coding=utf-8
import tensorflow as tf
a = tf.constant(10,name="a")
b = tf.constant(90,name="b")
y = tf.Variable(a+b*2,name='y')
init=tf.global_variables_initializer()
with tf.Session() as sess:
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('./log/a_plus_b',sess.graph)    
        sess.run(init)
        print(sess.run(y))

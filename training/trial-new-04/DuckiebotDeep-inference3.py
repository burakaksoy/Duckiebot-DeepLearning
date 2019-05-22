#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 22:09:05 2019

@author: burak
"""
import tensorflow as tf

# %% Main
if __name__ == "__main__":
    # %% Define the tensorflow placeholders
    # Placeholder for images 
    x = tf.placeholder(tf.float32, [1, 80, 160, 3], name="input")
    
    # %% Define tensorflow initializer
    initializer = tf.contrib.layers.xavier_initializer()    
    
    # %% Define tensorflow variables to be optimized
    # CNN Variables for weights and biases   
    W_1 = tf.Variable(initializer([7,7,3,16]),name='filters')
    W_1_0 = tf.Variable(tf.zeros([1], dtype="float32"))
    
    W_2 = tf.Variable(initializer([5,5,16,32]))
    W_2_0 = tf.Variable(tf.zeros([1], dtype="float32"))
    
    W_3 = tf.Variable(initializer([5,5,32,64])) 
    W_3_0 = tf.Variable(tf.zeros([1], dtype="float32"))
    
    W_4 = tf.Variable(initializer([5,5,64,64])) 
    W_4_0 = tf.Variable(tf.zeros([1], dtype="float32"))
    
    # Hidden layer weight and bias
    W_h = tf.Variable(initializer([5*10*64,1024]))
    W_h_0 = tf.Variable(tf.zeros([1,1024], dtype="float32"))
    
    # Output weight and bias
    W_o = tf.Variable(initializer([1024,2])) 
    W_o_0 = tf.Variable(tf.zeros([1,2], dtype="float32")) 
    
    # %% Define the tensorflow model
    # Batch size M
    M = tf.shape(x)[0]
#    M = tf.cast(M, tf.float32)
    
    ## Forward Propagation
    con1 = tf.math.add(tf.nn.conv2d(input=x,filter=W_1,strides=[1, 2, 2, 1],padding="SAME"), W_1_0)
    act1 = tf.nn.relu(con1) # Mx40x80x16
    
    con2 = tf.math.add(tf.nn.conv2d(input=act1,filter=W_2,strides=[1, 2, 2, 1],padding="SAME"), W_2_0)
    act2 = tf.nn.relu(con2) # Mx20x40x32
    
    con3 = tf.math.add(tf.nn.conv2d(input=act2,filter=W_3,strides=[1, 2, 2, 1],padding="SAME"), W_3_0)
    act3 = tf.nn.relu(con3) # Mx10x20x64
    
    con4 = tf.math.add(tf.nn.conv2d(input=act3,filter=W_4,strides=[1, 2, 2, 1],padding="SAME"), W_4_0)
    act4 = tf.nn.relu(con4) # Mx5x10x64
    
    # Flatten for hidden layer input
    X_in = tf.reshape(act4,[M,5*10*64]) # Mx3200
    # Calculate output of hidden layer
    H = tf.nn.relu(tf.matmul(X_in, W_h) + W_h_0) # Mx1024
    
    # Calculate output
    output = tf.matmul(H, W_o) + W_o_0 # Mx2
    # Store predict output
    predict_op = tf.identity(output, name="output")
    
    # %% TENSORFLOW RUN
    # start training
    saver = tf.train.Saver(tf.global_variables())
    # Session    
    with tf.Session() as session:
        # Initialize variables
        session.run(tf.global_variables_initializer())
        session.run(tf.local_variables_initializer())
    
        saver.restore(session, './my_model' )
        # %% T
        saver.save(session, './my_model-inference3' )
    
    
    
    
    
    
    
    
    
    
    
    
    
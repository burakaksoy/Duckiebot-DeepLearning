#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 23:12:17 2019

@author: Burak Aksoy
"""
import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os
import pickle
import time

def save_accuracy_n_loss(accuracy_mat,accuracy_mat_test,loss_vect,loss_vect2,filename):
    filehandler = open(filename,"wb")
    pickle.dump([accuracy_mat,accuracy_mat_test,loss_vect,loss_vect2], filehandler)
    filehandler.close()
# %% Main
if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath('__file__'))  # Current path
    imgs_dir = dir_path + "/data-combined/images"  # Train data images directory under current path
    dphi_dir = dir_path + "/data-combined"  # Corresponding d and phi directory under current path

    # %% Define useful parameters of the system
    image_width = 160 # Column
    image_height = 80 # Row
    image_channel = 3
    
    num_of_img = 9010 # Total data images
    num_of_test_img = 1010 # Validation data amount
    num_of_training_img = num_of_img - num_of_test_img # 8000
    
    n_rate = 0.0005 # Initial Learning rate hyper parameter
    batch_size = 200 # Training batch size hyper parameter
    max_epochs = 120 # Max Number of epochs
    
    # %% # Load images
    data = np.zeros([num_of_img,image_height,image_width,image_channel])  # will be whole data tensor without labels
    for i in range(1, num_of_img + 1, 1):
        filename = imgs_dir + "/" + str(i).zfill(5) + ".jpg"
        # Load the image
        image = mpimg.imread(filename)
    
        data[i - 1,:,:,:] = image
    del i, image, imgs_dir, dir_path
    # %% # Load Labels
    labels = np.zeros([num_of_img,2]) # will be d and phi
    filename = dphi_dir + "/d-and-phi.txt"
    with open(filename) as openfileobject:
        i = 0
        for line in openfileobject:
            labels[i,:] = line.split(",")
            i += 1
    del i, filename, line, dphi_dir
    
    # %%  Convert data to np.float32 and normalize by dividing 255
    data = data.astype(np.float32) /255.0
    labels = labels.astype(np.float32)  
    
    # %% Define validation data randomly from training data 
    order = np.random.permutation(num_of_img)
    order = order[0:num_of_test_img]
    
    test_data = data[order,:,:,:]
    test_labels = labels[order,:]
    
    # Delete the data chosen as test data from training data
    mask = np.ones(num_of_img,dtype=bool)
    mask[order] = False
    
    train_data = data[mask] 
    train_labels = labels[mask]
    del mask, order, data, labels
        
    # %% Define the tensorflow placeholders
    # Placeholder for images 
    x = tf.placeholder(tf.float32, [None, 80, 160, 3], name="input")
    # Placeholder for true d and phi 
    y = tf.placeholder(tf.float32, [None, 2])
    
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
    con1 = tf.math.add(tf.nn.conv2d(input=x,filter=W_1,strides=[1, 1, 1, 1],padding="SAME"), W_1_0)
    act1 = tf.nn.relu(con1)
    poo1 = tf.nn.pool(input=act1,window_shape=(2,2),pooling_type="MAX",padding="VALID",strides=(2,2)) # Mx40x80x16
    
    con2 = tf.math.add(tf.nn.conv2d(input=poo1,filter=W_2,strides=[1, 1, 1, 1],padding="SAME"), W_2_0)
    act2 = tf.nn.relu(con2)
    poo2 = tf.nn.pool(input=act2,window_shape=(2,2),pooling_type="MAX",padding="VALID",strides=(2,2)) # Mx20x40x32
    
    con3 = tf.math.add(tf.nn.conv2d(input=poo2,filter=W_3,strides=[1, 1, 1, 1],padding="SAME"), W_3_0)
    act3 = tf.nn.relu(con3)
    poo3 = tf.nn.pool(input=act3,window_shape=(2,2),pooling_type="MAX",padding="VALID",strides=(2,2)) # Mx10x20x64
    
    con4 = tf.math.add(tf.nn.conv2d(input=poo3,filter=W_4,strides=[1, 1, 1, 1],padding="SAME"), W_4_0)
    act4 = tf.nn.relu(con4)
    poo4 = tf.nn.pool(input=act4,window_shape=(2,2),pooling_type="MAX",padding="VALID",strides=(2,2)) # Mx5x10x64
    
    # Flatten for hidden layer input
    X_in = tf.reshape(poo4,[M,5*10*64]) # Mx3200
    # Calculate output of hidden layer
    H = tf.nn.relu(tf.matmul(X_in, W_h) + W_h_0) # Mx1024
    
    # Calculate output
    output = tf.matmul(H, W_o) + W_o_0 # Mx2
    # Store predict output
    predict_op = tf.identity(output, name="output")
    
    # Calculate average batch error
    err_mat = tf.math.abs(y - predict_op)
    err_avr = tf.reduce_mean(err_mat,axis=0) # d and phi averaga errors separetly
    # Define Loss
    # loss = tf.losses.mean_squared_error(y,predict_op)
    loss = (err_avr[0]/0.45) + (err_avr[1]/3.0)
    # Add the optimizer
    train_op = tf.train.AdamOptimizer(n_rate).minimize(loss)
        
    # %% TENSORFLOW RUN
    # start training
    saver = tf.train.Saver()
    # Session    
    with tf.Session() as session:
        # Initialize variables
        session.run(tf.global_variables_initializer())
        
        # Allocate memory for training performance evaluations
        loss_vect_train = []
        loss_vect_test = []
        err_mat_train = []
        err_mat_test = []
        
        # Start epochs
        for i_e in range(max_epochs):
            start_time = time.time()
            print("Epoch: " + str(i_e+1))
            
            ## Randomly choose a mini-batch of size batch_size from training data            
            # Shuffle the data
            data_order = np.random.permutation(num_of_training_img)
                        
            # Calculate number of batches for each epoch
            num_batches = int(num_of_training_img/batch_size)
#            num_batches = 2 # TODO: DELETE IT AFTER, DEBUG 
            for i in range(num_batches):
                print("Batch: " + str(i+1))

                # Select the data with chosen indexes
                X_batch = train_data[data_order[i*batch_size:(i+1)*batch_size],:,:,:]
                T_batch = train_labels[data_order[i*batch_size:(i+1)*batch_size],:]
                
                # Define the batch feed dictionary
                feed_dict_train = {x: X_batch, y: T_batch}
                
                # Store and Print train-accuracy, test-accuracy, Accuracy-per-class for test data, 
                # and loss value in each 54 iterations.
                if (i % 20 == 0) or ((i_e+1 == max_epochs) and (i+1 == num_batches)):
                    # Calculate train data mini-batch loss and accuracy
                    LOSS = session.run(loss, feed_dict = feed_dict_train)
                    ERR_train = session.run(err_avr, feed_dict = feed_dict_train)
                    
                    # Calculate test data loss and accuracy                    
                    LOSS2_lst = []
                    est_labels_lst = []
                    # Divide test data into smaller pieces for easier processing with CPU
                    test_batch_size = 202
                    for itr in range(0,num_of_test_img,test_batch_size):
                        # Define the test feed dictionary
                        feed_dict_test = {x: test_data[itr:itr+test_batch_size,:,:,:], y: test_labels[itr:itr+test_batch_size,:]}
                        # Calculate test data batch loss and accuracy 
                        LOSS2_val = session.run(loss, feed_dict = feed_dict_test)
                        # Estimate the labels
                        estimated_labels = session.run(predict_op, feed_dict = feed_dict_test)                        

                        # Append the calculated values to corresponding lists
                        LOSS2_lst.append(LOSS2_val)
                        est_labels_lst.append(estimated_labels)
                    # Convert lists to arrays and find the means for whole test data loss and accuracy
                    LOSS2 = np.array(LOSS2_lst)                    
                    LOSS2 = np.mean(LOSS2) # Test data loss
                    est_labels = np.array(est_labels_lst)
                    est_labels = np.reshape(est_labels,[num_of_test_img,2])
                    ERR_test = np.mean(np.absolute(test_labels - est_labels), axis=0)
                    
                    
                                      
                    print("Train Loss: "+str(LOSS)+",\nTest Loss: "+str(LOSS2)+",\nTrain Err(d,phi): "+str(ERR_train)+",\nTest Err(d,phi): "+str(ERR_test))
                    # Store the values in lists
                    loss_vect_train.append(LOSS)
                    loss_vect_test.append(LOSS2)
                    err_mat_train.append(ERR_train)
                    err_mat_test.append(ERR_test)
                
                # Update the weights and biases using the optimizer
                session.run(train_op, feed_dict = feed_dict_train)
            # %% # Decrease the learnin rate after each epoch
            n_rate = n_rate*0.90      
            finish_time = time.time()
            # Print epoch time
            print("Batch Size: " + str(batch_size) + ", Epoch Time: " + str(finish_time-start_time))
        # %% Training is done
        # this saver.save() should be within the same tf.Session() after the training is done
        # saver = tf.train.Saver(tf.global_variables())
        save_path = saver.save(session, "./my_model")
        # Convert lists to np.arrays
        err_mat_train = np.array(err_mat_train)
        err_mat_test = np.array(err_mat_test)
        
        loss_vect_train = np.array(loss_vect_train)
        loss_vect_test = np.array(loss_vect_test)
        # Save the accuracy values in a file
        file_name = "CNN_ERR-learnrate_"+str(n_rate)+".txt" 
        save_accuracy_n_loss(err_mat_train,err_mat_test,loss_vect_train,loss_vect_test,file_name)

    
    
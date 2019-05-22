#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 01:47:52 2019

@author: burak
"""

import pickle
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    
    
    with open('CNN_ERR-learnrate_2.7583690436774965e-06.txt', 'rb') as f:
        err_mat_train,err_mat_test,loss_vect_train,loss_vect_test = pickle.load(f)
        
# %%     
#    plt.figure()
#    plt.title("Average Pixel Error, Method: RNN", size = '15')
#    plt.plot(Num,err_vect_train, label='Training Data Error')
#    plt.plot(Num,err_vect_test, label='Test Data Error')
#    plt.xlabel('Epochs ', size = '15')
#    plt.ylabel(' Error', size = '15')
#    plt.grid(True)
#    plt.legend(prop={'size': 15})
#    plt.show()
    
# %%     
    t = np.arange(0.0, 70.1, 0.5)
    
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Epochs ', size = '15')
    ax1.set_ylabel('Error in V_left', color=color, size = '15')
    ax1.plot(t, err_mat_test[:141,0], linewidth=3.0, color=color)
    ax1.tick_params(axis='y',labelsize=15, labelcolor=color)
    ax1.tick_params(axis='x',labelsize=15)
    ax1.grid(True)
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    
    color = 'tab:blue'
    ax2.set_ylabel('Error in V_right', color=color, size = '15')  # we already handled the x-label with ax1
    ax2.plot(t, err_mat_test[:141,1], linewidth=3.0, color=color)
    ax2.tick_params(axis='y',labelsize=15, labelcolor=color)
    ax2.grid(True)
    
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    
    plt.title("Validation Data Error, End-to-end approach", size = '15')
    plt.show()
    
#    plt.figure()
#    plt.title("Loss Change Change, Method: RNN", size = '15')
#    plt.plot(Num,loss_vect_train, label='Training Data Loss')
#    plt.plot(Num,loss_vect_test, label='Test Data Loss')
#    plt.xlabel('Epochs ', size = '15')
#    plt.ylabel('Loss', size = '15')
#    plt.grid(True)
#    plt.legend(prop={'size': 15})
#    plt.show()
#    
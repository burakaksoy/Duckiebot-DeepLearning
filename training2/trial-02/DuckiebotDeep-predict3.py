#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 22:40:44 2019

@author: burak
"""

from mvnc import mvncapi as mvnc
import sys
import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os
import time

path_to_networks = './'
graph_filename = 'graph'

dir_path = os.path.dirname(os.path.realpath('__file__'))  # Current path
path_to_images = dir_path + "/../data-combined/images"  # Train data images directory under current path
image_filename = path_to_images + '/00416.jpg'

# %% Check Mividius Device 
#mvnc.SetGlobalOption(mvnc.GlobalOption.LOGLEVEL, 2)
devices = mvnc.EnumerateDevices()
if len(devices) == 0:
    print('No devices found')
    quit()

device = mvnc.Device(devices[0])
device.OpenDevice()

# %% Load graph
with open(path_to_networks + graph_filename, mode='rb') as f:
    graphfile = f.read()
# Send graph to the device
graph = device.AllocateGraph(graphfile)

# %% Read image Open CV
print('Reading image: ' + image_filename)
img = cv2.imread(image_filename)
# Convert color due to openCV defaults BGR to RGB
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#plt.imshow(img)
#plt.show()

## Read image mpimg
img2 = mpimg.imread(image_filename)
#plt.imshow(img2)
#plt.show()
cv2.imshow('image2',img2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Convert image to float16 and normalize so that device can process
img = img.astype(np.float16)/255.0

start = time.time()
# Send image to the device
print('Start download to NCS...')
graph.LoadTensor(img, 'user object')
output, userobj = graph.GetResult()

output = output.astype(np.float32)
print('Output(v_l,v_r): ' + str(output))
finish = time.time()
print('inference time: '+ str(finish-start))

# Clear and Disconnect Movidius device
graph.DeallocateGraph()
device.CloseDevice()
print('Finished')

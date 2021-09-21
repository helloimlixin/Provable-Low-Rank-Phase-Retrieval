#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 12:21:32 2021

@author: soominkwon
"""

import numpy as np
import matplotlib.pyplot as plt
from provable_lrpr import provable_lrpr_fit
from generate_lrpr import generateLRPRMeasurements


image_name = 'image_tensor_small.npz'
with np.load(image_name) as data:
    tensor = data['arr_0']
n1, n2, q_dim = tensor.shape

L = 5 # number of modulations
m_dim = n1 * n2 * L

images, Y, A = generateLRPRMeasurements(image_name=image_name, m_dim=m_dim, L=L)

X_hat = provable_lrpr_fit(Y=Y, A=A, rank=5, max_iters=10)

vec_first_image = X_hat[:, 0]

first_image = np.reshape(
    vec_first_image, (images.shape[0], images.shape[1]), order='F')

plt.imshow(np.abs(images[:, :, 0]), cmap='gray')
plt.title('True Image')
plt.show()

plt.imshow(np.abs(first_image), cmap='gray')
plt.title('Reconstructed Image via LRPR2')
plt.show()   

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 17:09:03 2023

@author: tommyfabian
"""

import numpy as np
import matplotlib.pyplot as plt


width, height = 1200,1200
xmin, xmax, ymin, ymax = -2.0, 2.0, -2.0, 2.0
        
x = np.linspace(xmin, xmax, width)
y = np.linspace(ymin, ymax, height)
X, Y = np.meshgrid(x, y)
Z = X + 1j * Y


c = np.zeros_like(Z)
mandelbrot_set = np.zeros_like(Z, dtype=float)


for i in range(100):
    c = c**2 + Z
    m = np.abs(c) < 2
    mandelbrot_set += m


plt.imshow(mandelbrot_set, cmap = 'gray')
plt.colorbar()
plt.show()

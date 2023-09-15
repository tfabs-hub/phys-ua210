#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 15:40:24 2023

@author: tommyfabian
"""

import numpy as np


def mc(L):
    M = 0.0
    for i in range(-L, L+1):
        for j in range(-L, L+1):
            for k in range(-L, L+1):
                if i == j == k == 0:
                    continue
                else:
                    M += ((-1)**(i+j+k))/np.sqrt(i**2 + j**2 + k**2)
    return M

  
M = mc(100)
print(f"The Madelung constant is {M}")





def mc2(L):
    x, y, z = np.ogrid[-L:L+1, -L:L+1, -L:L+1]
    
    distances = np.sqrt(x**2 + y**2 + z**2)

    distances[L, L, L] = np.inf
    
    madelung = np.sum((-1)**(np.abs(x+y+z)) / distances)
    
    return madelung

M2 = mc2(100)
print(f"The Madelung constant is {M2}")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 11:35:09 2023

@author: tommyfabian
"""

import numpy as np
import matplotlib.pyplot as plt


N = 10000
half_life = {'Bi213': 46 * 60, 'Pb209': 3.3 * 60, 'Tl209': 2.2 * 60}
prob_decay = {isotope: 1 - np.exp(-1 / half_life[isotope]) for isotope in half_life}
prob_route = {'Bi213': 0.9791, 'Pb209': 0.0209} 


state = {'Bi213': N, 'Pb209': 0, 'Tl209': 0, 'Bi209': 0}
history = {isotope: [state[isotope]] for isotope in state}


for t in range(20000):
    for isotope in ['Pb209', 'Tl209']:
        decayed = np.random.random(state[isotope]) < prob_decay[isotope]
        state[isotope] -= np.sum(decayed)
        state['Bi209'] += np.sum(decayed)

    decayed = np.random.random(state['Bi213']) < prob_decay['Bi213']
    state['Bi213'] -= np.sum(decayed)
    route = np.random.random(np.sum(decayed)) < prob_route['Bi213']
    state['Pb209'] += np.sum(route)
    state['Tl209'] += np.sum(~route)

    for isotope in state:
        history[isotope].append(state[isotope])


plt.figure(figsize=(10,6))
for isotope in history:
    plt.plot(history[isotope], label=isotope)
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Number of atoms')
plt.show()
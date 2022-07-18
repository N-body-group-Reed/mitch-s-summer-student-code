#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 10:17:07 2022

@author: uzair
"""

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

def update_plot(scatter, path, frame_number):
    global old_com
    
    try:
        data = np.load(path + '/' + str(frame_number * 100) + '.npy')
        pos = data[:, :3]
        masses = data[:, 6:7]
        masses = masses.reshape(masses.shape[0])
        
        com = np.sum(pos * masses.reshape(len(masses), 1), axis=0) / np.sum(masses)
        vel = com - old_com
        old_com = com
        print("Velocity:", vel, "Energy:", calc_energy(masses, pos, data[:, 3:6]))
        
        scatter._offsets3d = pos.T
        # print(data)
    except FileNotFoundError:
        pass

def create_plot(path, size=500):
   fig = plt.figure(figsize=(7,7))
   ax = plt.axes(xlim=(-size, size),ylim=(-size, size),zlim=(-size, size), projection='3d')
   scatter=ax.scatter(np.array([]), np.array([]))
   global anim 
   anim = FuncAnimation(fig, lambda f: update_plot(scatter, path, f), interval=0.1)
   plt.show()
    
def calc_energy(masses, pos, vel):
    KE = 0
    PE = 0
    for i in range(masses.shape[0]):
        KE += 0.5 * masses[i] * np.sum(vel[i] ** 2)
        ppe = 0
        for j in range(masses.shape[0]):
            if i != j:
                ppe -= masses[i] * masses[j] / np.sqrt(np.sum((pos[i] - pos[j]) ** 2))
                PE -= masses[i] * masses[j] / np.sqrt(np.sum((pos[i] - pos[j]) ** 2))
        print(ppe, 0.5 * masses[i] * np.sum(vel[i] ** 2))
    return KE + PE
    
if __name__ == '__main__':
    anim = None
    old_com = np.zeros(3)
    create_plot('animations/3_body/005')
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 10:17:07 2022

@author: uzair
"""

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

def update_plot(scatter, path, frame_number, t, masses):
    global old_com
    
    try:
        data = np.load(path + '/' + str(frame_number * 10) + '.npy')
        pos = data[:, :3]
        # print(pos)
        # print(data[0, 3:6])
        com = np.sum(pos * masses.reshape(len(masses), 1), axis=0) / np.sum(masses)
        vel = (com - old_com) / (t * 10)
        old_com = com
        
        # print(data[:, 3:6], calc_energy(masses, pos, data[:, 3:6], vel))
        print("Velocity:", vel, "Energy:", calc_energy(masses, pos, data[:, 3:6], vel))
        
        # scatter.set_offsets(pos)
        
        scatter._offsets3d = pos.T
        # print(data)
    except FileNotFoundError:
        pass
            
def create_plot(path, size=500):
   fig = plt.figure(figsize=(7,7))
   ax = plt.axes(xlim=(-size, size),ylim=(-size, size),zlim=(-size, size), projection='3d')
   scatter=ax.scatter(np.array([]), np.array([]))
   global anim 
   t, masses = load_anim(path)
   anim = FuncAnimation(fig, lambda f: update_plot(scatter, path, f, t, masses), interval=10)
   plt.show()
    
def calc_energy(masses, pos, vel, com_vel):
    KE = 0
    PE = 0
    for i in range(masses.shape[0]):
        KE += 0.5 * masses[i] * np.sum((vel[i]) ** 2)
        for j in range(masses.shape[0]):
            if i != j:
                PE -= masses[i] * masses[j] / np.sqrt(np.sum((pos[i] - pos[j]) ** 2))
    # print(KE, PE, end="\t")
    PE /= 2
    
    # KE = 0.5 * np.sum(np.sum( masses * (vel)**2 ))


    # # # Potential Energy:

    # # positions r = [x,y,z] for all particles
    # x = pos[:,0:1]
    # y = pos[:,1:2]
    # z = pos[:,2:3]

    # # matrix that stores all pairwise particle separations: r_j - r_i
    # dx = x.T - x
    # dy = y.T - y
    # dz = z.T - z

    # # matrix that stores 1/r for all particle pairwise particle separations 
    # inv_r = np.sqrt(dx**2 + dy**2 + dz**2)
    # inv_r[inv_r>0] = 1.0/inv_r[inv_r>0]

    # masses_T = masses.reshape((masses.shape[0], 1))

    # # sum over upper triangle, to count each interaction only once
    # PE = np.sum(np.sum(np.triu(-(masses*masses_T)*inv_r,1)))
    
    return KE + PE;

    
def load_anim(path):
    arr = np.load(path + '/data.npy')
    t = arr[0]
    masses = arr[1:]
    return t, masses

if __name__ == '__main__':
    anim = None
    old_com = np.zeros(3)
    create_plot('animations/3_body/000')
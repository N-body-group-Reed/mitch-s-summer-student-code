#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 10:17:07 2022

@author: uzair
"""

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


class NBodyView:
    def __init__(self, path, size):
        
        self.path = path
        self.size = size
        
        self.anim = None
        self.old_com = np.zeros(3)
        self.scatter = None
        self.t = None
        self.masses = None
        
        self.load_anim()
        self.create_plot()

    def update_plot(self, frame_number):
        global old_com
        try:
            data = np.load(self.path + '/' + str(frame_number * 10) + '.npy')
            pos = data[:, :3]
            com = np.sum(pos * self.masses.reshape(len(self.masses), 1), axis=0) / np.sum(self.masses)
            vel = (com - self.old_com) / (self.t * 10)
            self.old_com = com
            pos -= com
            
            # print(data[:, 3:6], calc_energy(masses, pos, data[:, 3:6], vel))
            # print(calc_energy(masses, pos, data[:, 3:6]))
            print("Velocity:", vel, "Energy:", self.calc_energy(pos, data[:, 3:6]))
            
            self.scatter.set_offsets(pos)
            
            # self.scatter._offsets3d = pos.T
            # print(data)
        except FileNotFoundError:
            # print(self.path + '/' + str(frame_number * 1) + '.npy')
            pass
                
    def create_plot(self):
       fig = plt.figure(figsize=(7,7))
       ax = plt.axes(xlim=(-self.size, self.size),ylim=(-self.size, self.size))#,
                     #zlim=(-self.size, self.size), projection='3d')
       
       self.scatter=ax.scatter(np.array([]), np.array([]))
       self.anim = FuncAnimation(fig, self.update_plot, interval=0.001)
       
       plt.show()
        
    def calc_energy(self, pos, vel):
        KE = 0
        PE = 0
        for i in range(self.masses.shape[0]):
            KE += 0.5 * self.masses[i] * np.sum((vel[i]) ** 2)
            for j in range(self.masses.shape[0]):
                if i != j:
                    PE -= self.masses[i] * self.masses[j] / np.sqrt(np.sum((pos[i] - pos[j]) ** 2))
        # print(KE, PE, end="\t")
        PE /= 2
        
        return KE + PE
        
    def load_anim(self):
        arr = np.load(self.path + '/data.npy')
        self.t = arr[0]
        self.masses = arr[1:]
        print(self.t)
    
    def isEnergyConserved(self, acceptableError=0.2):
        minEnergy = None
        maxEnergy = None
        
        for i in range(5000):
            data = np.load(self.path + '/' + str(i) + '.npy')
            pos = data[:, :3]
            # com = np.sum(pos * self.masses.reshape(len(self.masses), 1), axis=0) / np.sum(self.masses)
            
            e = self.calc_energy(pos, data[:, 3:6])
            if minEnergy == None or e < minEnergy:
                minEnergy = e
            if maxEnergy == None or e > maxEnergy:
                maxEnergy = e
        
        variability = (maxEnergy - minEnergy) / minEnergy
        
        return variability < acceptableError

if __name__ == '__main__':
    abc = NBodyView('animations/3_body_same_mass/000', 20)
    
    # for i in range(1000):
    #     if i % 50 == 0:
    #         print('-')
    #     if not isEnergyConserved('animations/3_body_same_mass/%03d' % i):
    #         print(i)
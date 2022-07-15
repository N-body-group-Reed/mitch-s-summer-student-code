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
    try:
        data = np.load(path + '/' + str(frame_number) + '.npy')
        pos = data[:, :3]
        scatter._offsets3d = pos.T
    except FileNotFoundError:
        pass

if __name__ == '__main__':
    fig = plt.figure(figsize=(7,7))
    size = 20000
    ax = plt.axes(xlim=(-size, size),ylim=(-size, size),zlim=(-size, size), projection='3d')
    scatter=ax.scatter(np.array([]), np.array([]))
    anim = FuncAnimation(fig, lambda f: update_plot(scatter, 'animations/orbit5', f), interval=0.1)
    plt.show() 
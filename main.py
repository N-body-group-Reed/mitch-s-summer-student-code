#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 09:29:36 2022

@author: uzair
"""

from nbody import *

# np.random.seed(1100)


numParticles = 2000
dist = 400
pos = np.zeros((numParticles + 1, 3))
velocity = np.zeros((numParticles + 1, 3))
masses = np.append(100000 * np.ones(1), 100 * np.ones(numParticles))
for i in range(numParticles):
    pos[i + 1] = dist * np.random.rand(3) - dist / 2
    pos[i + 1] = np.array([pos[i + 1][0], pos[i + 1][1], pos[i + 1][2] / 3])
    normal = pos[i + 1] / np.sqrt(np.sum(pos[i + 1] ** 2))
    velocity[i + 1] = np.sqrt(50 * 100000 / np.sqrt(np.sum(pos[i + 1] ** 2))) * np.array([-normal[1], normal[0], normal[2]])
    

# # # masses = 10 * np.random.rand(numParticles) + 1
# numParticles = 199
# pos = 400 * np.random.rand(numParticles, 3) + 200
# pos2 = -400 * np.random.rand(numParticles, 3) - 200
# pos = np.concatenate((np.array([[-400, -400, -400]]), np.array([[400, 400, 400]]), pos, pos2))

# velocity1 = -50 * np.random.rand(numParticles, 3)
# velocity2 = 50 * np.random.rand(numParticles, 3)
# velocity = np.concatenate((np.array([[0, 100, 0]]), np.array([[0, -100, 0]]), velocity1, velocity2))
# numParticles *= 2
# blackHoles = np.array([10000, 10000])
# masses = 20 * np.ones(numParticles)
# masses = np.append(blackHoles, masses)

# numParticles = 2
# pos = np.array([[0, 0, 0], [0, 100, 0]])
# velocity = np.array([[0, 0, 0], [0, 0, 0]])
# masses = np.array([1000, 1000])

# pos = np.array([[   0,   0, 0],
#                 [  0, 500 * 50 / 100, 0],
#                 [  0, 500 * 50 / 400, 0]])
# velocity = np.array([[ 0,  0,  0],
#                       [10, 0, 0],
#                       [0, 0, 20]])
# masses = np.array([500, 0.05, 0.05])

# oldAccel = np.zeros((numParticles, 3))
# G = 50 #6 * 10**(-11)
t = 0.05


nbody = NBody(pos, velocity, masses, 50)
saveFrames(nbody, 0.05, 'animations/orbit2', 2000)
# fig = plt.figure(figsize=(7,7))
# size = 400
# ax = plt.axes(xlim=(-size, size),ylim=(-size, size),zlim=(-size, size), projection='3d')
# # ax.set_title("Barnes Hut" if len(sys.argv) == 2 else "O(n^2)")
# # ax = fig.add_subplot(projection='3d')
# scatter=ax.scatter(pos[:,0], pos[:,1])#, pos[:, 2])
# anim = FuncAnimation(fig, updatePlot, interval=0.0001)
# # a = bh.centerOfMass(masses, pos).reshape((1, 3))
# # barnes_hut_nextTimeStep(masses, pos, velocity, oldAccel, t, G)
# # ax.scatter(a[:, 0], a[:, 1], a[:, 2], c='red')
# plt.show()

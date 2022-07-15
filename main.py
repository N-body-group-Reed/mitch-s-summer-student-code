#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 09:29:36 2022

@author: uzair
"""

from nbody import *
import numpy as np

# np.random.seed(1100)

def orbitCollision(numParticles, orbitDiameter):
    dist = orbitDiameter
    numParticles -= 2
    
    pos = np.zeros((numParticles + 2, 3))
    pos[0] = -dist * np.ones(3)
    pos[1] = dist * np.ones(3)
    velocity = np.zeros((numParticles + 2, 3))
    velocity[0] = np.array([50, 0, 0])
    velocity[0] = np.array([-50, 0, 0])
    masses = np.append(10000000 * np.ones(2), 1 * np.ones(numParticles))
    for i in range(1, numParticles // 2 + 1):
        pos[i + 1] = -dist * np.random.rand(3) - dist / 2
        pos[i + 1] = np.array([pos[i + 1][0], pos[i + 1][1], pos[i + 1][2]])
        normal = (pos[i + 1] - pos[0]) / np.sqrt(np.sum((pos[i + 1] - pos[0]) ** 2))
        velocity[i + 1] = np.sqrt(10000000 / np.sqrt(np.sum((pos[i + 1] - pos[0]) ** 2))) * np.array([-normal[1], normal[0], normal[2]])
    
    for i in range(numParticles // 2 + 1, numParticles + 1):
        pos[i + 1] = dist * np.random.rand(3) + dist / 2
        pos[i + 1] = np.array([pos[i + 1][0], pos[i + 1][1], pos[i + 1][2]])
        normal = (pos[i + 1] - pos[1]) / np.sqrt(np.sum((pos[i + 1] - pos[1]) ** 2))
        velocity[i + 1] = np.sqrt(10000000 / np.sqrt(np.sum((pos[i + 1] - pos[1]) ** 2))) * np.array([-normal[1], normal[0], normal[2]])
    
    nbody = NBody(pos, velocity, masses, 1)
    return nbody
    
def orbit(numParticles, orbitDiameter):
    dist = orbitDiameter
    pos = np.zeros((numParticles, 3))
    velocity = np.zeros((numParticles, 3))
    masses = np.append(1000000 * np.ones(2), 1 * np.ones(numParticles))
    for i in range(1, numParticles):
        pos[i + 1] = -dist * np.random.rand(3) - dist / 2
        pos[i + 1] = np.array([pos[i + 1][0], pos[i + 1][1], pos[i + 1][2]])
        normal = (pos[i + 1] - pos[0]) / np.sqrt(np.sum((pos[i + 1] - pos[0]) ** 2))
        velocity[i + 1] = np.sqrt(1000000 / np.sqrt(np.sum((pos[i + 1] - pos[0]) ** 2))) * np.array([-normal[1], normal[0], normal[2]])
    
    return NBody(pos, velocity, masses, 1)

def collision(dist=100, masses=np.array([1000, 1000])):
    numParticles = 2
    pos = np.array([[0, -dist/2, 0], [0, dist/2, 0]])
    velocity = np.array([[0, 0, 0], [0, 0, 0]])
    
    return NBody(pos, velocity, masses)

if __name__ == '__main__':
    saveFrames(orbitCollision(5000, 10000), t=0.05, 'animations/orbit5', 2000)

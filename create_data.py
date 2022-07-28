
from nbody import *
import random
import numpy as np

def create_fairly_complex_data():
    for i in range(1000):
        pos = 100 * np.random.rand(3, 3) - random.randint(0, 100)
        velocity = 40 * np.random.rand(3, 3) - 20
        mass = 5000 * np.random.rand(3)
        
        sim = NBody(pos, velocity, mass, 1),
        sim.save(0.0005, 'animations/3_body/%03d' % i, 5000, 60000, saveEvery=20)
        print("Generated Simulation #%03d" % i)
    
    

def create_super_simple_data():
    '''Generate 3 body systems that have equal masses 5and lie on a plane'''
    for i in range(1000, 2000):
        pos = 30 * np.random.rand(3, 3) - 15
        velocity = 4 * np.random.rand(3, 3) - 2
        
        pos[:, 2] = 0
        velocity[:, 2] = 0
        
        mass = 10 * np.ones(3)
        sim = NBody(pos, velocity, mass, 1, barnes_hut=False, use_collisions=False)
        sim.save(0.000025, 'animations/3_body_same_mass/%04d' % i, 5000, 5001, saveEvery=250)
        print("Generated Simulation #%04d" % i)

create_super_simple_data()
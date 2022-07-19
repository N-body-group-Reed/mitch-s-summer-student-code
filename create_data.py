
from nbody import *
import random

for i in range(1000):
    pos = 400 * np.random.rand(3, 3) - random.randint(0, 400)
    velocity = 6 * np.random.rand(3, 3) - 3
    mass = 1000000 * np.random.rand(3)
    saveFrames(NBody(pos, velocity, mass, 1), 0.00005, 'animations/3_body/%03d' % i, 20000, 60000)
    print("Generated Simulation #%03d" % i)
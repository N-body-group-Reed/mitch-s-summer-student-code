
from nbody import *
import random

for i in range(1000):
    pos = 100 * np.random.rand(3, 3) - random.randint(0, 100)
    velocity = 50 * np.random.rand(3, 3) - 25
    mass = 50000 * np.random.rand(3)
    saveFrames(NBody(pos, velocity, mass, 1), 0.00002, 'animations/3_body/%03d' % i, 5000, 60000, saveEvery=20)
    print("Generated Simulation #%03d" % i)
    
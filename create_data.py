
from nbody import *
import random

np.random.seed(17)
for i in range(1):
    pos = 100 * np.random.rand(3, 3) - random.randint(0, 100)
    velocity = 20 * np.random.rand(3, 3) - 10
    mass = 4000 * np.random.rand(3)
    saveFrames(NBody(pos, velocity, mass, 1), 0.002, 'animations/3_body/%03d' % i, 5000, 60000, saveEvery=25)
    print("Generated Simulation #%03d" % i)
    
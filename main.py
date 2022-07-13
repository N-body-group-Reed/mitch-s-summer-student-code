
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import barnes_hut as bh
import sys

### Position is stored as an N x 3 matrix 
### x1 y1 z1
### x2 y2 z2
### ...

def leapfrogIntegrate(pos, velocity, oldAccel, newAccel, t):
    '''Numerically integrates position and velocity based on a given acceleration and timestep'''

    halfStepVelocity = velocity + oldAccel * (t/2)
    nextPos = pos + halfStepVelocity * t
    nextVelocity = halfStepVelocity + newAccel * (t/2)
    return nextPos, nextVelocity

def calcNextTimeStep(masses, pos, velocity, oldAccel, t, G, softening=100):
    '''Basic O(n^2) method to update particle motion at a constant timestep'''
    accel = np.zeros((numParticles, 3))
    for i in range(numParticles):
        for j in range(numParticles):
            if i != j: 
                # find the difference between the two particles' positions
                diff = pos[j] - pos[i]

                # calculate the distance based on the vector between them
                distSquared = np.sum(diff ** 2) + softening

                # update acceleration based on the law of gravitation
                accel[i] += G * masses[j] * diff / (distSquared ** 1.5)

    pos, velocity = leapfrogIntegrate(pos, velocity, oldAccel, accel, t)
    return pos, velocity, accel

def barnes_hut_nextTimeStep(masses, pos, velocity, oldAccel, t, G, softening=100):
    '''Barnes-Hut Algorithm Implementation'''

    com = bh.centerOfMass(masses, pos)
    maxDist = np.max(np.sum((pos - com) ** 2, 1))
    # Create the tree structure
    root = bh.BarnesHutNode(com, maxDist)
    for i in range(numParticles):
        root.insert(pos[i], masses[i])
    
    # Calculate accelerations for each particle
    accel = np.zeros((numParticles, 3))
    
    for i in range(numParticles):
        accel[i] += G * bh.calcAcceleration(pos[i], root, 1, softening)
    
    # numerically integrate acceleration to update position and velocity
    pos, velocity = leapfrogIntegrate(pos, velocity, oldAccel, accel, t)
    return pos, velocity, accel

def updatePlot(frame_num, barnes_hut = True):
    '''Display the positions of each particle using matplotlib'''
    global pos, velocity, oldAccel, t, G
    # barnes_hut = len(sys.argv) != 1 and sys.argv[1] == "B"
    if barnes_hut:
        pos, velocity, oldAccel = barnes_hut_nextTimeStep(masses, pos, velocity, oldAccel, t, G)
    else:
        pos, velocity, oldAccel = calcNextTimeStep(masses, pos, velocity, oldAccel, t, G)  
    scatter._offsets3d = pos.T
    return (scatter)

if __name__ == '__main__':
    np.random.seed(1100)
    # numParticles = 10
    # pos = 400 * np.random.rand(numParticles, 3) - 200
    # velocity = 5 * np.random.rand(numParticles, 3)
    # # masses = 10 * np.random.rand(numParticles) + 1
    # masses = 20 * np.ones(numParticles)


    # numParticles = 2
    # pos = np.array([[90, 100, -20], [110, 100, 20]])
    # velocity = np.array([[0, 5, 0], [0, -5, 0]])
    # masses = np.array([50, 50])

    numParticles = 2
    pos = np.array([[   0,   0, 0],
                    [  0, 500 * 50 / 100, 0]])
    velocity = np.array([[ 0,  0,  0],
                          [10, 0, 0]])
    masses = np.array([500, 0.05])

    oldAccel = np.zeros((numParticles, 3))
    G = 50 #6 * 10**(-11)
    t = 0.01

    fig = plt.figure(figsize=(7,7))
    size = 400
    ax = plt.axes(xlim=(-size, size),ylim=(-size, size),zlim=(-size, size), projection='3d')
    # ax.set_title("Barnes Hut" if len(sys.argv) == 2 else "O(n^2)")
    # ax = fig.add_subplot(projection='3d')
    scatter=ax.scatter(pos[:,0], pos[:,1], pos[:, 2])
    anim = FuncAnimation(fig, updatePlot, interval=0.0001)
    # a = bh.centerOfMass(masses, pos).reshape((1, 3))
    # barnes_hut_nextTimeStep(masses, pos, velocity, oldAccel, t, G)
    # ax.scatter(a[:, 0], a[:, 1], a[:, 2], c='red')
    plt.show()

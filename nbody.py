
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import barnes_hut as bh



class NBody:
    def __init__(self, pos, vel, mass, G):
        ### Position is stored as an N x 3 matrix 
        ### x1 y1 z1
        ### x2 y2 z2
        ### ...
        
        self.numParticles = pos.shape[0]
        self.pos = pos
        self.velocity = vel
        self.mass = mass
        self.oldAccel = np.zeros(3)
        self.G = G
    
    
    def leapfrogIntegrate(self, newAccel, t):
        '''Numerically integrates position and velocity based on a given acceleration and timestep'''
    
        halfStepVelocity = self.velocity + self.oldAccel * (t/2)
        nextPos = self.pos + halfStepVelocity * t
        nextVelocity = halfStepVelocity + newAccel * (t/2)
        
        self.pos = nextPos
        self.velocity = nextVelocity
        self.oldAccel = newAccel

    def calcNextTimeStep(self, t, softening=100):
        '''Basic O(n^2) method to update particle motion at a constant timestep'''
        accel = np.zeros((self.numParticles, 3))
        for i in range(self.numParticles):
            for j in range(self.numParticles):
                if i != j: 
                    # find the difference between the two particles' positions
                    diff = self.pos[j] - self.pos[i]
    
                    # calculate the distance based on the vector between them
                    distSquared = np.sum(diff ** 2) + softening
    
                    # update acceleration based on the law of gravitation
                    accel[i] += self.G * self.mass[j] * diff / (distSquared ** 1.5)
    
        self.leapfrogIntegrate(accel, t)
    
    def barnes_hut_nextTimeStep(self, t, softening=100):
        '''Barnes-Hut Algorithm Implementation'''
    
        com = bh.centerOfMass(masses, self.pos)
        maxDist = np.max(np.sum((self.pos - com) ** 2, 1))
        # Create the tree structure
        root = bh.BarnesHutNode(com, maxDist)
        for i in range(self.numParticles):
            root.insert(self.pos[i], self.mass[i])
        
        # Calculate accelerations for each particle
        accel = np.zeros((self.numParticles, 3))
        
        for i in range(self.numParticles):
            accel[i] += self.G * bh.calcAcceleration(self.pos[i], root, 0.5, softening)
        
        # numerically integrate acceleration to update position and velocity
        self.leapfrogIntegrate(accel, t)

def updatePlot(frame_num, barnes_hut = True):
    '''Display the positions of each particle using matplotlib'''
    global nbody, t
    # barnes_hut = len(sys.argv) != 1 and sys.argv[1] == "B"
    if barnes_hut:
        nbody.barnes_hut_nextTimeStep(t)
    else:
        nbody.calcNextTimeStep(t)
    scatter._offsets3d = nbody.pos.T
    return (scatter)

if __name__ == '__main__':
    np.random.seed(1100)
    # numParticles = 10
    # pos = 400 * np.random.rand(numParticles, 3) - 200
    # velocity = 5 * np.random.rand(numParticles, 3)
    # # masses = 10 * np.random.rand(numParticles) + 1
    # masses = 20 * np.ones(numParticles)


    # # numParticles = 2
    # pos = np.array([[90, 100, -20], [110, 100, 20]])
    # velocity = np.array([[0, 5, 0], [0, -5, 0]])
    # masses = np.array([50, 50])

    pos = np.array([[   0,   0, 0],
                    [  0, 500 * 50 / 100, 0],
                    [  0, 500 * 50 / 400, 0]])
    velocity = np.array([[ 0,  0,  0],
                          [10, 0, 0],
                          [0, 0, 20]])
    masses = np.array([500, 0.05, 0.05])

    # oldAccel = np.zeros((numParticles, 3))
    # G = 50 #6 * 10**(-11)
    t = 0.05
    
    
    nbody = NBody(pos, velocity, masses, 50)

    fig = plt.figure(figsize=(7,7))
    size = 400
    ax = plt.axes(xlim=(-size, size),ylim=(-size, size),zlim=(-size, size), projection='3d')
    # ax.set_title("Barnes Hut" if len(sys.argv) == 2 else "O(n^2)")
    # ax = fig.add_subplot(projection='3d')
    scatter=ax.scatter(pos[:,0], pos[:,1])#, pos[:, 2])
    anim = FuncAnimation(fig, updatePlot, interval=0.0001)
    # a = bh.centerOfMass(masses, pos).reshape((1, 3))
    # barnes_hut_nextTimeStep(masses, pos, velocity, oldAccel, t, G)
    # ax.scatter(a[:, 0], a[:, 1], a[:, 2], c='red')
    plt.show()
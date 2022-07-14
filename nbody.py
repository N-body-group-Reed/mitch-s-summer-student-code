
import numpy as np
import barnes_hut as bh
import time

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
    
    def barnes_hut_nextTimeStep(self, t, softening=500):
        '''Barnes-Hut Algorithm Implementation'''
    
        com = bh.centerOfMass(self.mass, self.pos)
        maxDist = np.max(np.sum((self.pos - com) ** 2, 1))
        # Create the tree structure
        root = bh.BarnesHutNode(com, maxDist)
        for i in range(self.numParticles):
            root.insert(self.pos[i], self.mass[i])
        
        # Calculate accelerations for each particle
        accel = np.zeros((self.numParticles, 3))
        
        for i in range(self.numParticles):
            accel[i] += self.G * bh.calcAcceleration(self.pos[i], root, 1, softening)
        
        # numerically integrate acceleration to update position and velocity
        self.leapfrogIntegrate(accel, t)

def saveFrames(nbody, t, path, numFrames, numFramesPerNotification=5):
    '''Saves data from nbody model into animation frame files to be played back later'''
    t1 = time.time()
    start = t1
    for i in range(numFrames):
        with open(path + '/' + str(i) + '.npy', 'wb') as f:
            np.save(f, nbody.pos)
        nbody.barnes_hut_nextTimeStep(t)
        if (i + 1) % numFramesPerNotification == 0:
            t2 = time.time()
            print("Completed", i + 1, "frames!        Time per frame: %.2f s" %
                  ((t2 - t1) / numFramesPerNotification))
            t1 = time.time()
    end = time.time()        
    print("Simulation complete!")
    print("Generated", numFrames, "in %.2f seconds" % (end - start))
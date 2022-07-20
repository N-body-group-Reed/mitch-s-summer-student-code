
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
        self.justCollided = [False for i in range(self.numParticles)]
    
    
    def leapfrogKickDrift(self, t):
        halfStepVelocity = self.velocity + self.oldAccel * (t/2)
        nextPos = self.pos + halfStepVelocity * t
        
        self.pos = nextPos
        
    def leapfrogFinalKick(self, newAccel, t):
        halfStepVelocity = self.velocity + self.oldAccel * (t/2)
        nextVelocity = halfStepVelocity + newAccel * (t / 2)
        
        self.velocity = nextVelocity
        self.oldAccel = newAccel
    
    def leapfrogIntegrate(self, newAccel, t):
        '''Numerically integrates position and velocity based on a given acceleration and timestep'''
        
        halfStepVelocity = self.velocity + self.oldAccel * (t/2)
        nextPos = self.pos + halfStepVelocity * t
        nextVelocity = halfStepVelocity + newAccel * (t/2)
        
        self.pos = nextPos
        self.velocity = nextVelocity
        self.oldAccel = newAccel

    # def calcNextTimeStep(self, t, softening=0):
    #     '''Basic O(n^2) method to update particle motion at a constant timestep'''
    #     accel = np.zeros((self.numParticles, 3))
    #     for i in range(self.numParticles):
    #         for j in range(self.numParticles):
    #             if i != j: 
    #                 # find the difference between the two particles' positions
    #                 diff = self.pos[j] - self.pos[i]
    
    #                 # calculate the distance based on the vector between them
    #                 distSquared = np.sum(diff ** 2) + softening
    
    #                 # update acceleration based on the law of gravitation
    #                 accel[i] += self.G * self.mass[j] * diff / (distSquared ** 1.5)
    
    #     self.leapfrogIntegrate(accel, t)
    
    def barnes_hut_nextTimeStep(self, t):
        '''Barnes-Hut Algorithm Implementation'''
        
        self.leapfrogKickDrift(t)
        
        com = bh.centerOfMass(self.mass, self.pos)
        
        # set tree size based on the maximum dist from center of mass to any particle
        maxDist = np.max(np.sum((self.pos - com) ** 2, 1))
            
        # Create the tree structure
        root = bh.BarnesHutNode(com, maxDist)
        for i in range(self.numParticles):
            root.insert(self.pos[i], self.mass[i], self.velocity[i])
        
        # Calculate accelerations for each particle
        accel = np.zeros((self.numParticles, 3))
        collided = False
        
        for i in range(self.numParticles):
            if not self.justCollided[i]:
                accel[i] = self.G * bh.calcAcceleration(self.pos[i], self.mass[i], root, 1)
        # update position and velocity
        self.leapfrogFinalKick(accel, t)
        
        
        newVelocities = np.array(self.velocity)
        for i in range(self.numParticles):
            newVel, collided = bh.handle_elastic_collisions(self.pos[i], self.velocity[i], self.mass[i], root, 0.1)
            if not self.justCollided[i]:
                newVelocities[i] = newVel
            self.justCollided[i] = collided
        self.velocity = newVelocities
        
        # print(self.pos)
        # print(self.velocity)
        # print(self.oldAccel)
        # print()
        # self.leapfrogIntegrate(accel, t)
        

def saveFrames(nbody, t, path, numFrames, numFramesPerNotification=5, saveEvery=10):
    '''Saves data from nbody model into animation frame files to be played back later'''
    with open(path + '/' + 'data.npy', 'wb') as f:
        data = np.concatenate((np.array([t * saveEvery]), nbody.mass))
        np.save(f, data)
    
    t1 = time.time()
    start = t1
    for i in range(numFrames):
        with open(path + '/' + str(i) + '.npy', 'wb') as f:
            # combine position, velocity, and mass into a single array and save it
            data = np.concatenate((nbody.pos, nbody.velocity), axis=1)
            np.save(f, data)
        
        for j in range(saveEvery):
            nbody.barnes_hut_nextTimeStep(t)
        
        # print an update every few frames
        if (i + 1) % numFramesPerNotification == 0:
            t2 = time.time()
            print("Completed", i + 1, "frames!        Time per frame: %.2f s" %
                  ((t2 - t1) / numFramesPerNotification))
            t1 = time.time()
            
    end = time.time()        
    print("Simulation complete!")
    print("Generated", numFrames, "frames in %.2f seconds" % (end - start))
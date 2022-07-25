
import numpy as np
import barnes_hut as bh
import physics_helper as ph
import time

class NBody:
    def __init__(self, pos, vel, mass, G, barnes_hut=True, use_collisions=False,
                 softening=0):
        '''Implementation of an n-body system
        
        pos and vel are stored as an N x 3 matrix
        mass is a 1-dimensional numpy array'''
        
        self.numParticles = pos.shape[0]
        self.pos = pos
        self.velocity = vel
        self.mass = mass
        self.G = G
        self.barnes_hut = barnes_hut
        self.use_collisions = use_collisions
        self.softening = softening
        
        self.oldAccel = np.zeros(3)
        self.justCollided = [False for i in range(self.numParticles)]
        
    def leapfrogKickDrift(self, t):
        self.velocity += self.oldAccel * (t/2)
        self.pos += self.velocity * t
        
    def leapfrogFinalKick(self, newAccel, t):
        self.velocity += newAccel * (t / 2)
        self.oldAccel = newAccel
    
    def naive_step(self, t):
        '''Basic O(n^2) method to update particle motion'''
        
        self.leapfrogKickDrift(t)
        
        accel = np.zeros((self.numParticles, 3))
        newVelocities = np.array(self.velocity)
        for i in range(self.numParticles):
            for j in range(self.numParticles):
                if i != j: 
                    # find the difference between the two particles' positions
                    diff = self.pos[j] - self.pos[i]
    
                    # calculate the distance based on the vector between them
                    distSquared = np.sum(diff ** 2) + self.softening**2
    
                    # if this particle has just been in a collision, 
                    # that force will be much stronger than gravitational forces.
                    #
                    # This also prevents particles from getting pulled into each other 
                    # by giving them time to separate
                    if not self.justCollided[i]:
                        # update acceleration based on the law of gravitation
                        accel[i] += self.G * self.mass[j] * diff / (distSquared ** 1.5)
                    
                    # handle collisions
                    if self.use_collisions:
                        newVel = ph.elastic_collision(self.pos[i], self.vel[i], self.mass[i],
                                                      self.pos[j], self.pos[j], self.mass[j])[0]
                        collided = newVel != self.velocity[i]
                        if collided and not self.justCollided[i]:
                            newVelocities[i] = newVel
                            self.justCollided[i] = True
                            
                        self.justCollided[i] = collided
        
        self.leapfrogFinalKick(accel, t)
        
        if self.use_collisions:
            self.velocity = newVelocities
        
    def barnes_hut_step(self, t):
        '''Barnes-Hut Algorithm Implementation'''
        
        self.leapfrogKickDrift(t)
        
        # set tree size based on the maximum dist from center of mass to any particle
        com = ph.centerOfMass(self.mass, self.pos)
        maxDist = np.max(np.sum((self.pos - com) ** 2, 1)) + 1
            
        # Create the tree structure
        root = bh.BarnesHutNode(com, maxDist)
        for i in range(self.numParticles):
            root.insert(self.pos[i], self.mass[i], self.velocity[i])
            
        # Calculate accelerations for each particle
        accel = np.zeros((self.numParticles, 3))
        collided = False
        for i in range(self.numParticles):
            
            # if this particle has just been in a collision, 
            # that force will be much stronger than gravitational forces.
            #
            # This also prevents particles from getting pulled into each other 
            # by giving them time to separate
            if not self.justCollided[i]:
                accel[i] = self.G * bh.calcAcceleration(self.pos[i], self.mass[i], root, 0.5, self.softening)
                
        self.leapfrogFinalKick(accel, t)
        
        
        # Handle collisions
        if self.use_collisions:
            newVelocities = np.array(self.velocity)
            for i in range(self.numParticles):
                newVel = bh.handle_elastic_collisions(self.pos[i], self.velocity[i], self.mass[i], root, 0.5)
                collided = newVel != self.velocity[i]
                if collided and not self.justCollided[i]:
                    newVelocities[i] = newVel
                    self.justCollided[i] = True
                    
                self.justCollided[i] = collided
                
            self.velocity = newVelocities
        
    def advanceSimulation(self, t):
        if self.barnes_hut:
            self.barnes_hut_step(t)
        else:
            self.naive_step(t)
    
    def save(self, t, path, numFrames, numFramesPerNotification=5, saveEvery=10):
        '''Saves data from nbody model into animation frame files to be played back later
        
        If a variable time step is to be used, t represents the max possible time step allowed'''
        
        # store the timestep and mass in a data file since they are the same for
        # each frame
        with open(path + '/' + 'data.npy', 'wb') as f:
            data = np.concatenate((np.array([t * saveEvery]), self.mass))
            np.save(f, data)
        
        t1 = time.time()
        start = t1
        for i in range(numFrames):
            with open(path + '/' + str(i) + '.npy', 'wb') as f:
                # combine position and velocity into a single array and save it
                data = np.concatenate((self.pos, self.velocity), axis=1)
                np.save(f, data)
            
            # skip over frames that we aren't saving
            for j in range(saveEvery):
                self.advanceSimulation(t)
            
            # print an update every few frames
            if (i + 1) % numFramesPerNotification == 0:
                t2 = time.time()
                print("Completed", i + 1, "frames!        Time per frame: %.2f s" %
                      ((t2 - t1) / numFramesPerNotification))
                t1 = time.time()
                
        end = time.time()        
        print("Simulation complete!")
        print("Generated", numFrames, "frames in %.2f seconds" % (end - start))
    
    def calc_energy(self):
        '''Calculates the total energy of the current state of the n body system'''
        
        KE = 0
        PE = 0
        for i in range(self.numParticles):
            KE += 0.5 * self.mass[i] * np.sum((self.velocity[i]) ** 2)
            for j in range(self.numParticles):
                if i != j:
                    PE -= self.G * self.mass[i] * self.mass[j] / np.sqrt(np.sum((self.pos[i] - self.pos[j]) ** 2))
        # print(KE, PE, end="\t")
        PE /= 2
        
        return KE + PE
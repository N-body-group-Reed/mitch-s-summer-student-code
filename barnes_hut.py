
import numpy as np
import matplotlib.pyplot as plt

class BarnesHutNode:
    def __init__(self, center, width):
        self.spatialCenter = center # center of the node (cube-shaped) in space
        self.width = width # width of the node
        self.centerMass = np.zeros(3) # the center of mass of all particles contained in this node
        self.totalMass = 0 # total mass of particles in this node
        self.children = {} 
        self.isLeaf = True

    def insert(self, particle, particleMass):
        '''Inserts a particle into this node based on it's location'''
        self.totalMass += particleMass
        if self.isLeaf and self.totalMass == particleMass: # if this node doesn't have any particles in it
            self.centerMass = particle # place the particle in this node

        elif not self.isLeaf: # if this node has children
            # update center of mass
            self.centerMass = (self.centerMass * (self.totalMass - particleMass) + particle * particleMass) / self.totalMass

            self.insertInAppropriateChild(particle, particleMass)

        else: # if this node already contains a single particle

            # split this node up and assign particles to sub-nodes
            self.isLeaf = False
            self.insertInAppropriateChild(self.centerMass, self.totalMass - particleMass)
            self.insertInAppropriateChild(particle, particleMass)

            self.centerMass = (self.centerMass * (self.totalMass - particleMass) + particle * particleMass) / self.totalMass
            

    def insertInAppropriateChild(self, particle, particleMass):
        '''Inserts a node into the appropriate child node based on its position relative to the center of the node'''
        diff = particle - self.spatialCenter
        childName =  "+" if (diff[0] > 0) else "-"
        childName += "+" if (diff[1] > 0) else "-"
        childName += "+" if (diff[2] > 0) else "-"
        # Childname is in the format '[x dir][y dir][z dir]' where each [dir] is a + or - based 
        # on where the particle is relative to the node's center
        #
        # Ex: +++ means that the particle belongs in the child node that has a larger x, y, and z coord
        # relative to the current node's center

        # if the node that this particle belongs to is empty (so it isn't being stored)
        if childName not in self.children.keys():
            # create the node and store it in the children dictionary
            try:
                signs = diff / np.abs(diff)
            except:
                absDiff = np.abs(diff)
                absDiff[0] = 1 if absDiff[0] == 0 else absDiff[0]
                absDiff[1] = 1 if absDiff[1] == 0 else absDiff[0]
                absDiff[2] = 1 if absDiff[2] == 0 else absDiff[0]
                signs = diff / absDiff
                
            childCenter = signs * self.width / 4 + self.spatialCenter
            self.children[childName] = BarnesHutNode(childCenter, self.width/2)

        # insert the particle in the appropriate child node
        self.children[childName].insert(particle, particleMass)
    
def calcAcceleration(particle, node, threshold, softening=100):
    '''Calculate the net acceleration on a particle based on the Barnes-Hut algorithm'''
    accel = np.zeros(3)
    diff = node.centerMass - particle
    distSquared = np.sum(diff ** 2) + softening
    
    if node.isLeaf: # if the node only has one particle
        accel += node.totalMass * diff / (distSquared ** 1.5)
    else: # if the node contains multiple particles
        sd_ratio = node.width / np.sqrt(distSquared)
        if sd_ratio < threshold:
            # if the node is far away, treat all the particles within it as a single mass at its center
            accel += node.totalMass * diff / (distSquared ** 1.5)
        else: # if the node is nearby
            # Visit each childnode and determine its effects on this particle
            for child in node.children.values():
                accel += calcAcceleration(particle, child, threshold, softening)
    
    return accel

        

def centerOfMass(masses, pos):
    '''Calculate the center of mass of a group of particles'''
    com = np.sum(pos * masses.reshape(len(masses), 1), axis=0) / np.sum(masses)
    return com

if __name__ == '__main__':
    numParticles = 100
    pos = 200 * np.random.rand(numParticles, 3) - 100
    root = BarnesHutNode(centerOfMass(np.ones(numParticles), pos), 200)
    for i in range(numParticles):
        root.insert(pos[i], 1)
    fig = plt.figure(figsize=(7,7))
    ax = plt.axes(xlim=(-400, 400),ylim=(-400, 400),zlim=(-400, 400), projection='3d')
    scatter=ax.scatter(pos[:,0], pos[:,1], pos[:, 2])
    plt.show()


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
        self.velocity = np.zeros(3) # used for collisions; only used when node is a leaf

    def insert(self, particle, particleMass, vel):
        '''Inserts a particle into this node based on it's location'''
        self.totalMass += particleMass
        if self.isLeaf and self.totalMass == particleMass: # if this node doesn't have any particles in it
            self.centerMass = particle # place the particle in this node
            self.velocity = vel

        elif not self.isLeaf: # if this node has children
            # update center of mass
            self.centerMass = (self.centerMass * (self.totalMass - particleMass) + particle * particleMass) / self.totalMass

            self.insertInAppropriateChild(particle, particleMass, vel)

        else: # if this node already contains a single particle

            # split this node up and assign particles to sub-nodes
            self.isLeaf = False
            self.insertInAppropriateChild(self.centerMass, self.totalMass - particleMass, self.velocity)
            self.insertInAppropriateChild(particle, particleMass, vel)
            
            self.velocity = np.zeros(3)
            self.centerMass = (self.centerMass * (self.totalMass - particleMass) + particle * particleMass) / self.totalMass
            

    def insertInAppropriateChild(self, particle, particleMass, vel):
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
        self.children[childName].insert(particle, particleMass, vel)
    
def calcAcceleration(particle, mass, node, threshold):
    '''Calculate the net acceleration on a particle based on the Barnes-Hut algorithm'''
    accel = np.zeros(3)
    diff = node.centerMass - particle
    distSquared = np.sum(diff ** 2) + 0
    if node.isLeaf: # if the node only has one particle
        if distSquared != 0:
            accel += node.totalMass * diff / (distSquared ** 1.5)
            # accel += soften(particle, node.centerMass, mass, node.totalMass, np.sqrt(distSquared))
    else: # if the node contains multiple particles
        if distSquared == 0:
            sd_ratio = threshold + 1
        else:
            sd_ratio = node.width / np.sqrt(distSquared)
            
        if sd_ratio < threshold:
            # if the node is far away, treat all the particles within it as a single mass at its center
            
            accel += node.totalMass * diff / (distSquared ** 1.5)
            # accel += soften(particle, node.centerMass, mass, node.totalMass, np.sqrt(distSquared))
        else: # if the node is nearby
            # Visit each childnode and determine its effects on this particle
            for child in node.children.values():
                accel += calcAcceleration(particle, mass, child, threshold)
    
    return accel

def handle_elastic_collisions(particle, vel, mass, node, threshold, radius=12):
    '''Causes particles to bounce off of each other when they get close
    Returns new velocity, whether there was a collision'''
        
    diff = node.centerMass - particle
    dist = np.sqrt(np.sum(diff ** 2))
    if node.isLeaf: # if the node only has one particle
        if dist != 0:
            if dist < 2 * radius:
                # TODO
                # Figure out where this comes from:
                # https://physics.stackexchange.com/questions/681396/elastic-collision-3d-eqaution
                
                normal = diff / dist
                eff_mass = 1 / (1/mass + 1/node.totalMass)
                impact_speed = np.dot(normal, (vel - node.velocity))
                impulse_magnitude = 2 * eff_mass * impact_speed
                # print(particle, node.centerMass)
                # print(mass, node.totalMass)
                # print(vel, -normal * impulse_magnitude / mass)
                # print(node.velocity)
                # print()
                
                return vel + (-normal * impulse_magnitude / mass), True
    else: # if the node contains multiple particles
        if dist == 0:
            sd_ratio = threshold + 1
        else:
            sd_ratio = node.width / dist
            
        if sd_ratio < threshold:
            # if the node is far away, treat all the particles within it as a single mass at its center
            return vel, False
            # accel += node.totalMass * diff / (distSquared ** 1.5)
            # accel += soften(particle, node.centerMass, mass, node.totalMass, np.sqrt(distSquared))
        else: # if the node is nearby
            for child in node.children.values():
                v, collided = handle_elastic_collisions(particle, vel, mass, child, threshold, radius)
                if collided:
                    return v, True
    
    return vel, False
    
    

def centerOfMass(masses, pos):
    '''Calculate the center of mass of a group of particles'''
    com = np.sum(pos * masses.reshape(len(masses), 1), axis=0) / np.sum(masses)
    return com
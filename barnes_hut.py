
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
    
def calcAcceleration(particle, mass, node, threshold, softening=100):
    '''Calculate the net acceleration on a particle based on the Barnes-Hut algorithm'''
    accel = np.zeros(3)
    diff = node.centerMass - particle
    distSquared = np.sum(diff ** 2) + softening
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
                accel += calcAcceleration(particle, mass, child, threshold, softening)
    
    return accel

def soften(pos1, pos2, mass1, mass2, dist, radius = 60):
    '''Softens the force of particle 2 on particle 1 by treating particles as spheres with volume'''
    
    diff = pos2 - pos1
    if dist >= 2 * radius:
        return mass2 * diff / dist ** 3
    
    totalVol = 4/3 * np.pi * (radius ** 3)
    vol_non_intersect = np.pi * dist * (radius ** 2 - (dist ** 2) / 12)
    vol_intersect = totalVol - vol_non_intersect
    com_intersect = (pos1 - pos2) / 2
    com_non_intersect_p1 = (totalVol * pos1 - vol_intersect * com_intersect) / vol_non_intersect
    com_non_intersect_p2 = (totalVol * pos2 - vol_intersect * com_intersect) / vol_non_intersect

    percentIntersect = vol_intersect / totalVol
    percentNonintersect = vol_non_intersect / totalVol
    
    squared_dist_p2_nonintersecting_p1 = np.sum((pos2 - com_non_intersect_p1) ** 2)
    squared_dist_nonintersecting_p2_intersecting_p1 = np.sum((com_non_intersect_p2 - com_intersect) ** 2)
    
    accel_p2_nonintersecting_p1 = percentNonintersect * mass2 * diff / squared_dist_p2_nonintersecting_p1 ** 1.5
    accel_nonintersecting_p2_on_intersecting_p1 = percentIntersect * mass2 * percentNonintersect * diff / squared_dist_nonintersecting_p2_intersecting_p1 ** 1.5
    accel_repel = 100 * diff * (-dist + 2 * radius) / (mass1 * dist ** 2)
    
    accel = accel_p2_nonintersecting_p1 + accel_nonintersecting_p2_on_intersecting_p1 + accel_repel
    # print(accel, pos1, pos2, vol_intersect, vol_non_intersect, accel_nonintersecting_p2_on_intersecting_p1, accel_p2_nonintersecting_p1)
    return accel
    

def centerOfMass(masses, pos):
    '''Calculate the center of mass of a group of particles'''
    com = np.sum(pos * masses.reshape(len(masses), 1), axis=0) / np.sum(masses)
    return com
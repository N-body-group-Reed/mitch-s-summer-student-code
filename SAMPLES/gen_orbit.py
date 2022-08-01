import argparse
import numpy as np
import nbody.physics_helper as ph

def orbit(outfile, numParticles, orbitDiameter, bigMass, smallMass):
    '''Simulates small particles orbit around a massive particle'''
    
    pos = np.zeros((numParticles, 3))
    velocity = np.zeros((numParticles, 3))
    # Give one particle a much larger mass than all of the other particles
    masses = np.append(bigMass * np.ones(1), smallMass * np.ones(numParticles - 1))
    for i in range(1, numParticles):
        # randomly place particles some distance away from the center
        pos[i] = orbitDiameter * np.random.rand(3) - orbitDiameter / 2
        pos[i, 2] /= 3 # flatten out the orbit 
        
        # get the normal vector between center and particle
        normal = (pos[i] - pos[0]) / ph.dist(pos[i], pos[0])
        perpendicular_dir = np.array([-normal[1], normal[0], normal[2]])
        
        # tangential velocity = sqrt(centripetal acceleration * radius)
        # v = sqrt((large_mass / r^2) * r) = sqrt(large_mass / r) if G is 1
        vel_magnitude = np.sqrt(bigMass / ph.dist(pos[i], pos[0]))
        
        # make particle move fast enough to orbit in a direction tangent to the normal vector
        velocity[i] = vel_magnitude * perpendicular_dir
    
    data = np.concatenate((pos, velocity, masses.reshape(numParticles, 1)), 1)
    np.savetxt(outfile, data)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='A tool to generate random initial conditions for an orbit of a given size')
    parser.add_argument('outfile', help="The output file location containing the intial conditions")
    parser.add_argument('num_particles', help="The number of orbitting particles, including the massive particle in the center",
                        type=int, default=10)
    parser.add_argument('-r', '--max_radius', help="The maximum radius of the orbit",
                        type=int, default=100)
    parser.add_argument('-m', '--particle_mass', help="The mass of each orbitting particle",
                        type=int, default=5)
    parser.add_argument('-b', '--big_particle_mass', help="The mass of the center particle",
                        type=int, default=10000)

    args = parser.parse_args()
    
    orbit(args.outfile, args.num_particles, args.max_radius * 2, args.big_particle_mass, args.particle_mass)

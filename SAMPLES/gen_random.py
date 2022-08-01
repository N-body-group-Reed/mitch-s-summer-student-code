import argparse
import numpy as np

def random_ic(outfile, numParticles, pos_range, vel_range, max_mass):
    '''Simulates small particles orbit around a massive particle'''
    
    pos = 2 * pos_range * np.random.rand(numParticles, 3) - pos_range
    velocity = 2 * vel_range * np.random.rand(numParticles, 3) - vel_range
    masses = max_mass * np.random.rand(numParticles)
    
    data = np.concatenate((pos, velocity, masses.reshape(numParticles, 1)), 1)
    np.savetxt(outfile, data)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='A tool to generate random initial conditions')
    parser.add_argument('outfile', help="The output file location containing the intial conditions")
    parser.add_argument('num_particles', help="The number of particles",
                        type=int, default=10)
    parser.add_argument('-p', '--pos_range', help="All particles will be located in the range (-pos_range, pos_range)",
                        type=int, default=50)
    parser.add_argument('-v', '--vel_range', help="All velocities will be in the range (-vel_range, vel_range)",
                        type=int, default=20)
    parser.add_argument('-m', '--max_mass', help="The maximum possible mass",
                        type=int, default=5000)

    args = parser.parse_args()
    
    random_ic(args.outfile, args.num_particles, args.pos_range, args.vel_range, args.max_mass)

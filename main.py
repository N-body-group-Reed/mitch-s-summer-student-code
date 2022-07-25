#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 11:45:59 2022

@author: uzair
"""

import nbody
import nbody_view
import argparse

parser = argparse.ArgumentParser(description='An N-Body Simulator')
parser.add_argument("input_file", help="file containing initial conditions(pos, vel, mass) of all particles")
parser.add_argument("output_dir", help="folder to store simulation output")
parser.add_argument("time", help="the length of time that will be simulated", type=int)
parser.add_argument("-b", "--barnes_hut", help="Use the barnes-hut algorithm to speed up execution",
                    action="store_true")
parser.add_argument("-c", "--collide", help="Allow particles to collide elastically",
                    action="store_true")

args = parser.parse_args()


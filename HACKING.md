# HACKING

## What This File Is

This is a quick start guide to using the command line interface of this package. It will explain all of the available options that you can use to get the most out of this tool. This file will be using the sample initial conditions available in the SAMPLES directory of this repo. 

## The Basics

The CLI has two modes: simulation and visualization.
Let's start by simulating an orbit from the SAMPLES directory.

First, we need to create a folder to store our simulation output. This can be done in Linux as:
```bash
$ mkdir simulation_output/
```

To create a simulation using the default parameters, go to your command line and type:
```bash
$ gravbody S SAMPLES/orbit10.txt simulation_output 10
```
The `S` tells the program that it will be running in simulation mode. We are passing in the initial conditions from the SAMPLES directory and telling the program to output the simulation results in the simulation_output directory. The `10` signifies that this simulation will run for 10 time units.

Now to view the results of this simulation, type:
```bash
$ gravbody V simulation_output
```

You should now see a particles orbitting! 

## Simulation Options

To view all of the simulation options, type:
```bash 
$ gravbody S -h
```

We won't go over all of the options here, but there are a few worth mentioning.

### Time step

The `time_step` option allows you to control how often the simulation updates the particles' motions. A smaller time step will result in a more accurate simulation, but will cause the simulation to take longer and generate more files. 

Here is an example:
```bash
$ gravbody S SAMPLES/orbit10.txt simulation_output 10 --time_step 0.005
```

### Store Every n Frames

If you look in the simulation_output directory now, you should see that the number of frames has now doubled. Before modifying the timestep, there were 1000 files, but now there are 2000. Since we are updating the particle positions more frequently, we have more data. 

But, what if you don't want that much data? Is it possible to keep the accuracy of a lower time step without generating so many files? That's what the `save_every` option is for! 
Let's start by clearing out the simulation_data directory:
```bash
$ rm simulation_data/*
```

Now run:
```bash
$ gravbody S SAMPLES/orbit10.txt simulation_output 10 --time_step 0.005 --save_every 2
```

There should now be 1000 files in the simulation_output directory. Normally, there would be 2000 since the simulation runs for 10 time units / 0.005 time units per step. But, since we set the save_every option to 2, the simulation is saving only 1 frame for every 2 frames that it generates (saves every other frame), causing us to have half the number of frames. This results in a higher accuracy without having to store more data. Keep in mind that `time_step` represents the amount of time between simulation updates, not necessarily the amount of time between each outputted frame.

### Collisions

Currently, particles can be pulled into each other if they get too close. This can cause issues with our simulation, as the closer the particles get to each other, the faster they go. When particles get really, really close together, the simulation can work in unexpected ways, causing particles to fly off at very high velocities. One way to fix this is to reduce the time step, but this causes the whole simulation to take much longer. Another solution is to have particles bounce off of each other when they start to get close. 

Here is an example:

```bash
$ gravbody S SAMPLES/rand3.txt simulation_output 15 -c 5
```

The `-c` tells the program that we want to use collisions. The `5` represents the radius of each particle. If you run the view command (at the top of this file), you should see the particles bouncing off of each other. 


## Visualization Options

To view all of the simulation options, type:
```bash 
$ gravbody V -h
```

We won't go over all of the options here, but there are a few worth mentioning.

### Resize

To set the size of the view, we can use the `size` option. The visualization will show the plot with all axes having coordinates between (-size, size).
For example: 
```bash
$ gravbody V simulation_output --size 1000
```
This will set the plot to show x, y, and z coordinates between -1000 and 1000. 
You can also zoom in or out on the plot while it is running by right clicking on the plot and dragging your mouse. 

### Time Scale

Some simulations may have low time steps, resulting in the visualization to update very slowly. You can set the `time_scale` value to skip a number of frames every visualization update. 

```bash
$ gravbody V simulation_output --time_scale 2
```
The command above will display every other frame, making the simulation go by twice as fast. 

### Energy Plot

One feature of the visualization tool is that it can plot the energy of the system over time so that you can verify that it remains mostly constant. Here's an example:
```bash
$ gravbody V simulation_output -e
```


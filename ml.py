#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 14:20:04 2022

@author: uzair
"""

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


class NBodyDataSet(Dataset):
    def __init__(self, root_dir, startAnim=0, endAnim=900, numFrames=5000):
        self.numAnims = endAnim - startAnim
        self.numFrames = numFrames
        self.start = startAnim
        self.end = endAnim
        self.root_dir = root_dir
    def __len__(self):
        return self.numAnims * self.numFrames
    def __getitem__(self, idx):
        anim = idx // (self.numFrames)
        frame = (idx % (self.numFrames))
        first = np.load("%s/%03d/0.npy" % (self.root_dir, anim + self.start))   
        
        data = np.load("%s/%03d/data.npy" % (self.root_dir, anim + self.start))
        masses = data[1:]
        t_step = data[0]
        
        expected = np.load("%s/%03d/%d.npy" % (self.root_dir, anim + self.start, frame))
        t = t_step * frame
        
        masses_T = masses.reshape((masses.shape[0], 1))
        
        input_data = first#np.concatenate((first, masses_T), axis=1)
        return torch.from_numpy(np.append(input_data.flatten(), t)), torch.from_numpy(expected.flatten())

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        # self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(19, 19),
            nn.ReLU(),
            nn.Linear(19, 18),
            nn.ReLU(),
            nn.Linear(18, 18)
        )

    def forward(self, x):
        # x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def train_one_epoch(epoch_index):
    running_loss = 0
    last_loss = 0
    for i, data in enumerate(training_loader):
        inpt, expected = data
        
        optimizer.zero_grad()
        outputs = model(inpt.float())
        # print(inpt)
        # print(outputs)
        # print(expected)
        # print('\n')
        loss = loss_fn(outputs.float(), expected.float())
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 10 == 9:
            last_loss = running_loss / 10 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            running_loss = 0.
            
    return last_loss

training_set = NBodyDataSet('animations/3_body_same_mass', 0, 990, 5000)
validation_set = NBodyDataSet('animations/3_body_same_mass', 990, 1000, 5000)
 
training_loader = DataLoader(training_set, batch_size=5000, shuffle = True, num_workers = 8)
validation_loader = DataLoader(validation_set, batch_size=5000, shuffle = False, num_workers = 3)

loss_fn = torch.nn.MSELoss()
model = NeuralNetwork().to(device)  
model.load_state_dict(torch.load("models/model_20220724_145959_18"))
optimizer = torch.optim.SGD(model.parameters(), lr=0.000001, momentum=0.001) 

def train():
    print(model)
    
    
    # Initializing in a separate cell so we can easily add more epochs to the same run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    epoch_number = 0
    
    EPOCHS = 5000
    
    best_vloss = 1_000_000_000.
    
    for epoch in range(2, EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))
    
        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(epoch_number)
    
        # We don't need gradients on to do reporting
        model.train(False)
    
        running_vloss = 0.0
        vinputs, voutputs = None, None
        for i, vdata in enumerate(validation_loader):
            vinputs, vexpected = vdata
            voutputs = model(vinputs.float())
            vloss = loss_fn(voutputs.float(), vexpected.float())
            running_vloss += vloss
        
        print(vinputs, voutputs)
        
        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
    
        # Log the running loss averaged per batch
        # for both training and validation      
    
        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = 'models/model_{}_{}'.format(timestamp, epoch_number)
            torch.save(model.state_dict(), model_path)
    
        epoch_number += 1


def test():
    global anim
    model = NeuralNetwork()
    model.load_state_dict(torch.load("models/model_20220722_095529_35"))
    model.eval()
    size = 20
    fig = plt.figure(figsize=(7,7))
    ax = plt.axes(xlim=(-size, size),ylim=(-size, size),zlim=(-size, size), projection='3d')
    scatter=ax.scatter(np.array([]), np.array([]))
    
    path = 'animations/3_body_same_mass/075'
    
    first = np.load(path + '/0.npy')
    data = np.load(path + '/data.npy')
    t_step = data[0]
    mass = data[1:]
    mass_T = mass.reshape(mass.shape[0], 1)
    
    combined = first.flatten()#np.concatenate((first, mass_T), axis=1).flatten()
    def update_plot(frame_number):
        nonlocal model, scatter, combined, t_step
        inputs = np.append(combined.flatten(), frame_number * t_step * 10)
        prediction = model(torch.from_numpy(inputs).float())
        prediction = prediction.reshape(3, 6)
        pos_predicted = prediction[:, :3]
        # print(inputs)
        # print(prediction)
        # print(pos_predicted)
        # print()
        
        scatter._offsets3d = pos_predicted.T.detach().numpy()
        # scatter.set_offsets(pos_predicted.T.detach().numpy())
        
    anim = FuncAnimation(fig, update_plot, interval=0.1)
    plt.show() 

anim = None
train() 
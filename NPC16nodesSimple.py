#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 14:36:41 2021

@author: maria
"""


import numpy as np
from scipy.integrate import odeint
from scipy.linalg import circulant
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

print("hello world")
def pol2cart(rho, phi):
    '''Function transforms polar to cartesian coordinates'''
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def hexadecagonspring(y, t, Lrest, la, K, ka, fext, d, n):

    x0x, x0y, x1x, x1y, x2x, x2y, x3x, x3y, x4x, x4y, x5x, x5y, x6x, x6y, x7x, x7y, x8x, x8y, x9x, x9y, x10x, x10y, x11x, x11y, x12x, x12y, x13x, x13y, x14x, x14y, x15x, x15y, v0x, v0y, v1x, v1y, v2x, v2y, v3x, v3y, v4x, v4y, v5x, v5y, v6x, v6y, v7x, v7y, v8x, v8y, v9x, v9y, v10x, v10y, v11x, v11y, v12x, v12y, v13x, v13y, v14x, v14y, v15x, v15y = y

    
    x = np.array([[x0x, x0y], [x1x, x1y], [x2x, x2y], [x3x, x3y], [x4x, x4y], [x5x, x5y], [x6x, x6y], [x7x, x7y],
                  [x8x, x8y], [x9x, x9y], [x10x, x10y], [x11x, x11y], [x12x, x12y], [x13x, x13y], [x14x, x14y], [x15x, x15y]]) 
    v = np.array([[v0x, v0y], [v1x, v1y], [v2x, v2y], [v3x, v3y], [v4x, v4y], [v5x, v5y], [v6x, v6y], [v7x, v7y],
                  [v8x, v8y], [v9x, v9y], [v10x, v10y], [v11x, v11y], [v12x, v12y], [v13x, v13y], [v14x, v14y], [v15x, v15y]]) 

    anc = np.array([0., 0.]) # spacial coordinates of anchor node 
          
    allaccarray = np.zeros((16,2)) # array for accelerations of node 0 - 15
    
    for i in range(16): # i indicates the reference node 
        
        accarray = np.array([0., 0.]) # initiate acceleration array for each node i 
        
        for j in [k for k in range(-n, n+1) if k != 0]: # j is neighbour nodes -n to +n relative to i, skipping 0 (0=i)
            
            jnew = (i+j)%16 
            accarray += K[i][jnew]  * (x[i]-x[jnew])/np.linalg.norm(x[i]-x[jnew]) * (Lrest[i][jnew] - np.linalg.norm(x[i]-x[jnew]))

        accarray += ka * (x[i] - anc)/np.linalg.norm(x[i] - anc) * (la - np.linalg.norm(x[i] - anc)) #anchor
        accarray = fext[i] + accarray - d*v[i] # external force and damping
        allaccarray[i] = accarray 

    dxdt = np.concatenate((v.flatten(), allaccarray.flatten()))                                                                
    return dxdt

### lengths of the spring
la = 1 # radius hexadecagon (= length to anchor); set to 1

# Generate cartesian coordinates of the Hexadecagon using its polar coordinates 
angle = 0.
cartc = np.zeros(32)

for i in range(0,32,2): # skip every other entry to populate it with y-coords
    x, y = pol2cart(la,angle)
    cartc[i] = x
    cartc[i+1] = y
    angle += 0.125*np.pi
    
cart2D = cartc.reshape(16,2) # reshape so that x and y coords are in separate columns. TODO 
le = np.linalg.norm(cart2D[0,:]-cart2D[1,:]) # distance between node 0 and 1
l2 = np.linalg.norm(cart2D[0,:]-cart2D[2,:])
l3 = np.linalg.norm(cart2D[0,:]-cart2D[3,:])
l4 = np.linalg.norm(cart2D[0,:]-cart2D[4,:])
l5 = np.linalg.norm(cart2D[0,:]-cart2D[5,:])
l6 = np.linalg.norm(cart2D[0,:]-cart2D[6,:])
l7 = np.linalg.norm(cart2D[0,:]-cart2D[7,:])
ld = np.linalg.norm(cart2D[0,:]-cart2D[8,:])

Lrest = circulant([0, le, l2, l3, l4, l5, l6, l7, ld, l7, l6, l5, l4, l3, l2, le])


### constants of springs. numbers correspond to the numbering in lengths 
ke = k2 = k3 = k4 = k5 = k6 = k7 = kd = 1 # spring constants 
K = circulant([0, ke, k2, k3, k4, k5, k6, k7, kd, k7, k6, k5, k4, k3, k2, ke])
ka = 0.3

# Other parameters
d = 1 # damping 
n = 6 # maximum distant neighbour to connect on each side 

# External forces 
fext = np.array([[5,0.]     ,   [0.,0.] ,   [0.,0.] ,   [0.,0.],
                 [0.0,0.]   ,   [0.,0.] ,   [0.,0.] ,   [0.,0.],
                 [-5.5,0.]    ,   [0.,0.] ,   [0.,0.] ,   [0.,0.],
                 [0.0,0.]   ,   [0.,0.] ,   [0.,0.] ,   [0.,0.]])



# starting values and timepoints 
y0 = np.concatenate((cartc, np.zeros(32))) # last 32 entries are starting velocities 
t_end = 50 # end timepoint
t = np.linspace(0, t_end, t_end*12)

sol16 = odeint(hexadecagonspring, y0, t, args=(Lrest, la, K, ka, fext, d, n)) #TODO: try out adaptive timesteps 
solplot = sol16


#### Plotting ####################################################################
fig, axs = plt.subplots(2, 1 ,figsize = (10, 15))
axs = axs.ravel()

# x and y position over time 
label = ["x0x", "x0y", "x1x", "x1y", "x2x", "x2y", "x3x", "x3y", "x4x", "x4y", "x5x", "x5y", "x6x", "x6y", "x7x", "x7y",
         "x8x", "x8y", "x9x", "x9y", "x10x", "x10y", "x11x", "x11y", "x12x", "x12y", "x13x", "x13y", "x14x", "x14y", "x15x", "x15y"]
for i in range(32):
    axs[0].plot(t, solplot[:, i], label = label[i])
axs[0].set(xlabel = 't')
axs[0].legend(loc = 'best')

axs[1].plot(solplot[-1, np.append([i for i in range(0,32,2)],0)], solplot[-1, np.append([i for i in range(1,32,2)],1)], 
 
  linestyle = ":", marker = "o", color="gray", markerfacecolor = "black", markersize = 10)

colourcode = True
if (colourcode ==False):
    #trajectory
    for i in range(0,32,2):
        axs[1].plot(solplot[:,i], solplot[:,i+1], color = "blue", linestyle = "-")
    
    axs[1].axis("scaled")
    
    axs[1].set(xlabel = "x", ylabel = "y")
    
    plt.tight_layout()

else:
    ### colourcoding velocities
    #summarize velocities in x and y direction into one velocity 
    pos = sol16[:,:32]
    speeds = sol16[:,32:]
    normspeeds = np.zeros((np.shape(speeds)[0],16))
    
    for node in range(0,31,2):    
        for step in range(np.shape(speeds)[0]):
            normspeeds[step,int(0.5*node)] = np.linalg.norm(speeds[step,node] - speeds[step,node+1])
    
    # plot trajectory
    #fig, axs = plt.subplots(1, 1, sharex=True, sharey=True)
    
    norm = plt.Normalize(normspeeds.min(), normspeeds.max()) 
    
    #####trajectory colorcoded for velocity
    for i in range(0,31,2):  
        points = pos[:,(i,i+1)].reshape(-1,1,2) # for node 1
        segments = np.concatenate([points[:-1],points[1:]], axis = 1)
        
    
        lc = LineCollection(segments, cmap = 'viridis', norm=norm)
        lc.set_array(normspeeds[:,int(0.5*i)])
        lc.set_linewidth(2)
        line = axs[1].add_collection(lc)
        
        axs[1].set_xlim(-5, 5)
        axs[1].set_ylim(-5, 5)
    #fig.colorbar(line, ax=axs[1])    
    
    axs[1].axis("scaled")
    
    axs[1].set(xlabel = "x", ylabel = "y")
    plt.tight_layout()
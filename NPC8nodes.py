#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 14:36:41 2021

@author: maria
"""


import numpy as np
#from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from scipy.linalg import circulant
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
#import seaborn as sns
import timeit

def pol2cart(rho, phi):
    '''Function transforms polar to cartesian coordinates'''
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def octagonspring(t, y, Lrest, la, K, ka, randf, d, n):
    ''''''
    x0x, x0y, x1x, x1y, x2x, x2y, x3x, x3y, x4x, x4y, x5x, x5y, x6x, x6y, x7x, x7y, v0x, v0y, v1x, v1y, v2x, v2y, v3x, v3y, v4x, v4y, v5x, v5y, v6x, v6y, v7x, v7y = y 
    x = np.array([[x0x, x0y], [x1x, x1y], [x2x, x2y], [x3x, x3y], [x4x, x4y], [x5x, x5y], [x6x, x6y], [x7x, x7y]]) 
    v = np.array([[v0x, v0y], [v1x, v1y], [v2x, v2y], [v3x, v3y], [v4x, v4y], [v5x, v5y], [v6x, v6y], [v7x, v7y]]) 

    anc = np.array([0., 0.]) # spacial coordinates of anchor node   
    symmetry = 8
    F = np.zeros((symmetry,2)) # Forces
    
##### TODO: test 

    for i in range(symmetry):
        F[i] = randf[i]*x[i] / np.linalg.norm([x[i], anc])

######    
      
    allaccarray = np.zeros((symmetry,2)) # array for accelerations of node 0 - 7
    
    for i in range(symmetry): # i indicates the reference node        
        accarray = np.array([0., 0.]) # initiate acceleration array for each node i 
        
        for j in [k for k in range(-n, n+1) if k != 0]: # j is neighbour nodes -n to +n relative to i, skipping 0 (0=i)            
            jnew = (i+j)%symmetry 
            accarray += K[i][jnew]  * (x[i]-x[jnew])/np.linalg.norm(x[i]-x[jnew]) * (Lrest[i][jnew] - np.linalg.norm(x[i]-x[jnew]))

        accarray += ka * (x[i] - anc)/np.linalg.norm(x[i] - anc) * (la - np.linalg.norm(x[i] - anc)) #anchor
        accarray = F[i] + accarray - d*v[i]  # external force and damping
        allaccarray[i] = accarray 

    dxdt = np.concatenate((v.flatten(), allaccarray.flatten()))                                                                
    return dxdt

# Generate cartesian coordinates of the octagon using its polar coordinates 
la = 1. # radius octagon (= length to anchor); set to 1
angle = 0.
cartc = np.zeros(16) 

for i in range(0,16,2): # skip every other entry to populate it with y-coords
    x, y = pol2cart(la,angle)
    cartc[i] = x
    cartc[i+1] = y
    angle += 0.25*np.pi

#cartc = np.array([1.,0,np.sqrt(2)/2, np.sqrt(2)/2,0,1,-np.sqrt(2)/2,np.sqrt(2)/2,-1,0,-np.sqrt(2)/2,-np.sqrt(2)/2,0,-1,np.sqrt(2)/2,-np.sqrt(2)/2]) #TODO remove

cart2D = cartc.reshape(8,2) # reshape so that x and y coords are in separate columns.  

le = np.linalg.norm(cart2D[0,:]-cart2D[1,:]) # distance between node i and i+1
l2 = np.linalg.norm(cart2D[0,:]-cart2D[2,:])
l3 = np.linalg.norm(cart2D[0,:]-cart2D[3,:])
ld = np.linalg.norm(cart2D[0,:]-cart2D[4,:])

Lrest = circulant([0., le, l2, l3, ld, l3, l2, le])

### spring constants. indices correspond to the indices of spring lengths 
ke = k2 = k3 = 1 #TODO: Pick smarter spring constants? 
kd = 0.5 # diagonal spring at 0.5 k, because there will be 2 of them (ktotal = k1+k2 for parallel springs)
K = circulant([0., ke, k2, k3, kd, k3, k2, ke])
ka = 0.5

# Other parameters
d = 1 # damping 
n = 2 # n maximum distant neighbour to connect on each side 

### Forces
finalmag = 1 # total magnitude
randf = np.zeros(8) # magnitude of radial forces acting on nodes 0-8

for i in range(8): 
    #randf[i] = np.random.exp()
    randf[i] = np.random.normal(0,0.1)

#Determine total magnitude of distortions
initialmag = sum(abs(randf))
if(initialmag != 0): randf = finalmag/initialmag * randf

# starting coordinates and velocities of nodes 
y0 = np.concatenate((cartc, np.zeros(16))) # last 16 entries are starting velocities 

start = timeit.default_timer()

# Solve ODE
sol8 = solve_ivp(octagonspring, [0,100], y0, method='RK45', args=(Lrest, la, K, ka, randf, d, n))

stop = timeit.default_timer()
print('Time: ', stop - start) 

#### Plotting ####################################################################
colourcode = True # Trajectories colourcoded by velocity, or monochrome. 
solplot = sol8.y.T # solutions for plotting
tplot = sol8.t # timesteps for plotting
#ax = sns.heatmap(solplot) 
plt.rcParams.update({'font.size': 25})
fig, axs = plt.subplots(2, 1, figsize = (13, 20))
#plt.title("magnitude: " + str(finalmag) + " damping: " + str(d) + " n: " + str(n))
axs = axs.ravel()

# x and y position over time 
label = ["x0", "y0", "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4", "x5", "y5", "x6", "y6", "x7", "y7"]
for i in range(16):
    axs[0].plot(tplot, solplot[:, i], label = label[i])
axs[0].set(xlabel = 't (a.u.)')
axs[0].legend(loc = 'best')


## Nodes at final position with trajectory and connecting springs 
# Just the Nodes
axs[1].plot(solplot[-1, [i for i in range(0,16,2)]], solplot[-1, [i for i in range(1,16,2)]],  
   linestyle = "", marker = "o", color="gray", markerfacecolor = "black", markersize = 25, zorder = 50)

# Anchor, and connection to anchor
axs[1].plot([0,0], [0,0], marker = "o", color = "lightgray", markersize = 15)
for i in range(0,16,2):
    axs[1].plot((solplot[-1,i],0),(solplot[-1,i+1],0),
    linestyle = ":", marker = "", color="lightgray")

# Connecting springs. n is the number of neighbours that are connected both cw or ccw
if (n >= 1):
    axs[1].plot(solplot[-1, np.append([i for i in range(0,16,2)],0)], 
                solplot[-1, np.append([i for i in range(1,16,2)],1)],  
                linestyle = ":", marker = "", color="gray")

if (n >= 2):
    axs[1].plot(solplot[-1, (0,4,8,12,0)], 
                solplot[-1, (1,5,9,13,1)], linestyle = ":", marker = "", color = "gray") 
    
    axs[1].plot(solplot[-1, (2,6,10,14,2)], 
                solplot[-1, (3,7,11,15,3)], linestyle = ":", marker = "", color = "gray") 

if (n >= 3):
     axs[1].plot(solplot[-1, (0,6,12,2,8,14,4,10,0)], 
                 solplot[-1, (1,7,13,3,9,15,5,11,1)], linestyle = ":", marker = "", color = "gray")   

if (n == 4):
    for i in range(0,7,2):
         axs[1].plot(solplot[-1, (i,   i+8)], 
                     solplot[-1, (i+1, i+9)], linestyle = ":", marker = "", color="gray")   

if (colourcode==False):
    #trajectory
    for i in range(0,16,2):
        axs[1].plot(solplot[:,i], solplot[:,i+1], color = "blue", linestyle = "-")
    
    axs[1].axis("scaled")    
    axs[1].set(xlabel = "x (a.u.)", ylabel = "y (a.u.)")    
    plt.tight_layout()

else:
    ### colourcoding velocities
    pos = solplot[:,:16] # positions over time
    vel = solplot[:,16:] # velocities over time
    normvel = np.zeros((np.shape(vel)[0],8)) #shape: [steps,nodes]
    
    for node in range(0,15,2):    
        for step in range(np.shape(vel)[0]):
             normvel[step,int(0.5*node)] = np.linalg.norm([vel[step,node], vel[step,node+1]])
    
    norm = plt.Normalize(normvel.min(), normvel.max()) 
    
    #####trajectory colorcoded for velocity
    for i in range(0,15,2):  
        points = pos[:,(i,i+1)].reshape(-1,1,2) # for node i
        segments = np.concatenate([points[:-1],points[1:]], axis = 1)
           
        lc = LineCollection(segments, cmap = 'viridis', norm=norm, zorder = 100)
        lc.set_array(normvel[:,int(0.5*i)])
        lc.set_linewidth(3)
        line = axs[1].add_collection(lc)

    axcb = fig.colorbar(line, ax=axs[1])   
    axcb.set_label('velocity (a.u.)')
    
    axs[1].axis("scaled")
    
    axs[1].set(xlabel = "x (a.u.)", ylabel = "y (a.u.)")
    plt.tight_layout()
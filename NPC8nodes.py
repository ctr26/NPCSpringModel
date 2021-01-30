#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 14:36:41 2021

@author: maria
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import circulant
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from warnings import warn
import timeit

### Parameters
symmetry = 10
d = 0.5 # damping 
n = 10 # n maximum distant neighbour to connect on each side 
magnitude = 0
### 

def NPC(t, y, Lrest, la, K, ka, randf, d, n, symmetry = 8):
    ''''''
    v = np.reshape(y[2*symmetry:], (symmetry, 2))
    x = np.reshape(y[:2*symmetry], (symmetry, 2))

    anc = np.array([0., 0.]) # coordinates of anchor node   
    F = np.zeros((symmetry, 2)) # Forces
    
    for i in range(symmetry): # TODO test
        F[i] = randf[i]*x[i] / np.linalg.norm([x[i], anc])

    allaccarray = np.zeros((symmetry, 2)) # array for accelerations of node 0 - 7
    
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


def Pol2cart(rho, phi):
    '''Function transforms polar to cartesian coordinates'''
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def Initialcoords(la, symmetry = symmetry, angleoffset = 0): 
# Generate cartesian coordinates of the octagon using its polar coordinates #TODO real-data coordinates
    angle = 0.
    cartc = np.zeros(2*symmetry) 
    
    for i in range(0,2*symmetry,2): # skip every other entry to populate it with y-coords
        x, y = Pol2cart(la,angle)
        cartc[i] = x
        cartc[i+1] = y
        angle += 2*np.pi/symmetry
    cartc2D = cartc.reshape(symmetry,2)
    return cartc, cartc2D

la = 1
la2 = 1.05
cartc, cart2D = Initialcoords(la = la, angleoffset = 0)
cartc2, cart22D = Initialcoords(la = la2, angleoffset = 2*np.pi/(3*symmetry))
#cartc = np.array([1.,0,np.sqrt(2)/2, np.sqrt(2)/2,0,1,-np.sqrt(2)/2,np.sqrt(2)/2,-1,0,-np.sqrt(2)/2,-np.sqrt(2)/2,0,-1,np.sqrt(2)/2,-np.sqrt(2)/2]) #TODO remove

def Springlengths(cart2D, symmetry): # TODO generalize to different symmetries 
        l = np.zeros(int(np.floor(symmetry/2)))
        for i in range(len(l)):
            l[i] = np.linalg.norm(cart2D[0,:]-cart2D[(i+1),:])
        
        if(symmetry%2 == 0): #if symmetry is even
            Lrest = circulant(np.append(0, np.append(l, np.flip(l[:-1]))))
        else: # if symmetry is odd
            Lrest = circulant(np.append(0, [l, np.flip(l)]))
        return Lrest

def Springconstants(symmetry):
    k = np.ones(int(np.floor(symmetry/2)))
    if(symmetry%2 == 0): #if symmetry is even
        k[-1] = 0.5
        K = circulant(np.append(0, np.append(k, np.flip(k[:-1]))))
    else:
        K = circulant(np.append(0, [k, np.flip(k)]))
    return K

def Forces(dist = ("normal", "exp"), rings = 2, magnitude = 1):
    finalmag = magnitude # total magnitude
    randf = np.zeros(symmetry) # magnitude of radial forces acting on nodes 0-8
    randf2 = np.zeros(symmetry)
    
    if (dist == "normal"):
        for i in range(symmetry): randf[i] = np.random.normal(0,0.1)
    elif (dist == "exp"):
        for i in range(symmetry): randf[i] = np.random.exponential()
    
    if (rings == 2):
            for i in range(len(randf2)):
                randf2[i] = randf[i] + np.random.normal(0, 0)
    
    #Determine total magnitude of distortions
    initialmag = sum(abs(randf)) + sum(abs(randf2))
    if(initialmag != 0): 
        randf = finalmag/initialmag * randf
        randf2 = finalmag/initialmag * randf2
    return randf, randf2

Lrest = Springlengths(cart2D, symmetry = symmetry)
Lrest2 = Springlengths(cart22D, symmetry = symmetry)
K = Springconstants(symmetry = symmetry)
ka = 0.5

if(n > symmetry/2):
    n = int(np.floor(symmetry/2))
    warn("Selected number of neighbours n too large. n has been set to " + str(n) + ".")



randf, randf2 = Forces("normal", magnitude = magnitude)


# starting coordinates and velocities of nodes 
y0 = np.concatenate((cartc, np.zeros(2*symmetry))) # last 16 entries are starting velocities 
y02 = np.concatenate((cartc2, np.zeros(2*symmetry)))

start = timeit.default_timer()

# Solve ODE
sol = solve_ivp(NPC, [0,100], y0, method='RK45', args=(Lrest, la, K, ka, randf, d, n, symmetry))
sol2 = solve_ivp(NPC, [0,100], y02, method='RK45', args=(Lrest2, la2, K, ka, randf2, d, n, symmetry))

stop = timeit.default_timer()
print('Time: ', stop - start) 

#### Plotting ####################################################################
colourcode = True # Trajectories colourcoded by velocity, or monochrome. 
solplot = sol.y.T # solutions for plotting
tplot = sol.t # timesteps for plotting
#ax = sns.heatmap(solplot) 
plt.rcParams.update({'font.size': 25})
fig, axs = plt.subplots(2, 1, figsize = (13, 20))
#plt.title("magnitude: " + str(finalmag) + " damping: " + str(d) + " n: " + str(n))
axs = axs.ravel()

solplot2 = sol2.y.T
solplot3 = np.reshape(solplot,(len(solplot),2*symmetry,2))

# x and y position over time 
label = ["x", "y"]
for i in range(2*symmetry):
    axs[0].plot(tplot, solplot[:, i], label = label[i%2] + str(i))
axs[0].set(xlabel = 't (a.u.)')
#axs[0].legend(loc = 'best')


def plotting(solplot, n = n, colourcode = True, colourbar = True, mainmarkercolor = "black", symmetry = symmetry):
    solplot2D = np.reshape(solplot,(len(solplot),2*symmetry,2))
    
    # Nodes at last timestep
    axs[1].plot(solplot2D[-1, :symmetry, 0], solplot2D[-1,:symmetry,1], 
    linestyle = "", marker = "o", color="gray", markerfacecolor = mainmarkercolor, markersize = 25, zorder = 50)
    
    # Anchor springs
    axs[1].plot([0,0], [0,0], marker = "o", color = "lightgray", markersize = 15)
    for i in range(0, symmetry):
        axs[1].plot((solplot2D[-1,i,0], 0), (solplot2D[-1,i,1], 0),
        linestyle = ":", marker = "", color="lightgray")   
        
    # Radial springs TODO: fix 
    for ni in range(1, n+1):
        if(symmetry%ni == 0): # if the total number of nodes is a multiple of ni, the nodes will be devided into ni graphs
            for i in range(ni):
                connect = np.append([j for j in range(i, symmetry, ni)], i)
                axs[1].plot(solplot2D[-1, connect, 0],solplot2D[-1, connect, 1], linestyle = ":", marker = "", color="gray")
        else: # otherwise, all nodes are connected by one path TODO: WRONG!
            connectopen = np.array([i%symmetry for i in range(0, ni*symmetry, ni)])
            rep = int(len(connectopen)/len(set(connectopen)))
            if (rep==1):
                connect = np.append(connectopen, 0)
                axs[1].plot(solplot2D[-1, connect, 0],solplot2D[-1, connect, 1], linestyle = ":", marker = "", color="gray")
            else:
                print(rep)
                for k in range(rep):
                    connect = np.append([i%symmetry for i in range(k, int(ni*symmetry/rep), ni)], k)
                    axs[1].plot(solplot2D[-1, connect, 0],solplot2D[-1, connect, 1], linestyle = ":", marker = "", color="red")

 
    if (colourcode): # Colourcoded trajectory
        ### colourcoding velocities
        pos = solplot[:,:2*symmetry] # positions over time
        vel = solplot[:,2*symmetry:] # velocities over time
        normvel = np.zeros((np.shape(vel)[0], symmetry)) #shape: [steps,nodes]
        
        for node in range(0, 2*symmetry-1, 2):    
            for step in range(np.shape(vel)[0]):
                 normvel[step,int(0.5*node)] = np.linalg.norm([vel[step,node], vel[step,node+1]])
        
        norm = plt.Normalize(normvel.min(), normvel.max()) 
        
        #####trajectory colorcoded for velocity
        for i in range(0, 2*symmetry-1, 2):  
            points = pos[:,(i,i+1)].reshape(-1,1,2) # for node i
            segments = np.concatenate([points[:-1],points[1:]], axis = 1)
               
            lc = LineCollection(segments, cmap = 'viridis', norm=norm, zorder = 100)
            lc.set_array(normvel[:,int(0.5*i)])
            lc.set_linewidth(3)
            line = axs[1].add_collection(lc)
    
        if(colourbar):
            axcb = fig.colorbar(line, ax=axs[1])   
            axcb.set_label('velocity (a.u.)')
        
        axs[1].axis("scaled")
        
        axs[1].set(xlabel = "x (a.u.)", ylabel = "y (a.u.)")
        plt.tight_layout()  
        
    else: # monochrome trajectory
        for i in range(0,symmetry,2):
            axs[1].plot(solplot2D[:,i,0], solplot[:,i,1], color = "blue", linestyle = "-")
        
        axs[1].axis("scaled")    
        axs[1].set(xlabel = "x (a.u.)", ylabel = "y (a.u.)")    
        plt.tight_layout()                  
   

plotting(solplot, n = n, symmetry = symmetry)
#plotting(solplot2, n = n, colourbar = False, mainmarkercolor="darkblue", symmetry = symmetry)

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
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

### Parameters
symmetry = 8 # finegraining with multiples of 8 possible 
d = 1 # damping 
n = 3 # n maximum distant neighbour to connect on each side 
magnitude = 10 # Magnitude of distortion 
rings = 2 # Number of rings
ka = 0.5 # Spring constant anchor springs
corneroffset = 2*np.pi/(3.8*symmetry) + np.random.normal(0,0) # TODO: Number a rough estimate adapted from SMAP code. Research needed. 
ringoffset = np.pi/(3*symmetry) + np.random.normal(0,0) # TODO: Number a rough estimate adapted from SMAP code. Research needed. 
###  

if(n > symmetry/2):
    n = int(np.floor(symmetry/2))
    warn("Selected number of neighbours n too large. n has been set to " + str(n) + ".")
    

def NPC(t, y, Lrest, la, K, ka, randf, d, n, symmetry = 8):
    '''
    t: time points 
    y: values of the solution at t 
    Lrest: Circulant matrix of resting lengths of all springs 
    K: Circulant matrix of all radial spring constants 
    ka: Spring constants of anchor springs 
    randf: array of forces (length = symmetry) to be applied in radial direction to each node 
    d: Damping factor 
    n: Number of connected neighbours in cw and ccw direction for each node 
    symmetry (default: 8): Number of nodes 
    output: solutions at t. x and y components of positions and velocities of each node for each time-step 
    '''
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

def Initialcoords(la, symmetry = symmetry, corneroffset = 0, ringoffset = 0): #TODO real-data coordinates
    '''
    Generates cartesian coordinates of the NPC given radius and symmetry 
    ## Input ##
    la: NPC Radius
    symmetry: Number of corners
    angleoffset (rad): Rotates the NPC by the given offset. Default 0
    ## Return values ##
    Cartesian coordinates in 1D and 2D array format 
    '''
    angle = 0.
    cartc = np.zeros(2*symmetry) 
    
    for i in range(0, 2*symmetry, 2): # skip every other entry to populate it with y-coords
        x, y = Pol2cart(la, angle+corneroffset+ringoffset)
        cartc[i] = x
        cartc[i+1] = y
        angle += 2*np.pi/symmetry
    return cartc


def Springlengths(cartc, symmetry): 
    '''Compute lengths of springs from coordinates and returns circulant matrix
    '''
    cart2D = cartc.reshape(symmetry,2)
    l = np.zeros(symmetry)
    for i in range(len(l)):
        l[i] = np.linalg.norm(cart2D[0, :] - cart2D[i, :])      
    return circulant(l)


def Springconstants(symmetry): # TODO: More variable spring constants? 
    "Returns circulant matrix of spring constants "
    k = np.ones(int(np.floor(symmetry/2)))
    if(symmetry%2 == 0): #if symmetry is even
        k[-1] = k[-1]/2 # springs that connect opposite corners will be double-counted. Spring constant thus halved 
        K = circulant(np.append(0, np.append(k, np.flip(k[:-1]))))
    else: #if symmetry is odd 
        K = circulant(np.append(0, [k, np.flip(k)]))
    return K


def Forces(dist = ("normal", "exp"), corneroffset = corneroffset, rings = 2, magnitude = 1, symmetry = symmetry):
    '''Returns array of Forces that are later applied in radial direction to the NPC corners
    ## Input ## 
    dist: Distribution to sample forces from. options: normal, or exponential distribution
    rings: Number of Rings to apply forces to. Forces are correlated between rings. Default is 2. 
    magnitude: Total magnitude of distortion. Default is 1. 
    ## Return ## 
    1D array of forces applied to each node 
    '''
    np.random.seed(0)
    randf = np.zeros(symmetry) # magnitude of radial forces acting on nodes 0-8
    randf2 = np.zeros(symmetry)
    
    if (dist == "normal"):
        for i in range(symmetry): randf[i] = np.random.normal(0, 1)
    elif (dist == "exp"):
        for i in range(symmetry): randf[i] = np.random.exponential()    
    
    if (rings == 1):
        initialmag = sum(abs(randf))       
        
    elif (rings == 2):
        angle = 2*np.pi/symmetry
        weight1 = (angle - corneroffset)/angle# iversely weights closeness of ring2 node i to ring1 node i... 
        weight2 = corneroffset/angle # ...and  to ring1 node i+1 
        
        for i in range(symmetry): 
            randf2[i] = weight1 * randf[i] + weight2 * randf[(i+1)%symmetry] + np.random.normal(0, 0)
            #randf2[i] = randf[i] #TODO: delete
        initialmag = 0.5 * (sum(abs(randf)) + sum(abs(randf2)))
        
    if(initialmag != 0): 
        randf = magnitude/initialmag * randf
        randf2 = magnitude/initialmag * randf2
    return randf, randf2

def ForcesMultivariateNorm(cartc, cartc2, symmetry, magnitude = 1): # TODO: include distances to nucleoplasmic ring 
    ''''''
    
    cartall = np.append(cartc, cartc2)
    cartall = cartall.reshape(2*symmetry, 2)
  
    AllL = np.zeros((symmetry*2, symmetry*2))
    
    for i in range(symmetry*2):
        for j in range(symmetry*2):
            AllL[i, j] = np.linalg.norm(cartall[i, :] - cartall[j, :])
    
    #AllL = circulant(l)
    mean = list(np.zeros(symmetry*2))
    #mean = (0, 0,0)
    LInv = AllL.max() - AllL
    
    cov = []
    for i in range(symmetry*2):
        cov.append(list(LInv[i]/AllL.max()))
            
    rng = np.random.default_rng()
    F = rng.multivariate_normal(mean, cov)
    
    initialmag = sum(F)
    
    if (initialmag != 0):
        F = magnitude/initialmag * F
          
    return F[0:symmetry], F[symmetry:]

  

la = 50 # TODO: Number a rough estimate adapted from SMAP code. Research needed. 
la2 = 54 # TODO: Number a rough estimate adapted from SMAP code. Research needed. 


randf, randf2 = Forces("normal", magnitude = magnitude, rings = rings)


def returnparameters(la, corneroffset = 0, ringoffset = 0, symmetry = symmetry):
    cartc = Initialcoords(la = la, corneroffset=corneroffset, ringoffset=ringoffset)
    Lrest = Springlengths(cartc, symmetry)
    K = Springconstants(symmetry)
    y0 = np.concatenate((cartc, np.zeros(2*symmetry))) # starting coordinates and velocities of nodes. last half of the entries are starting velocities 
    return cartc, Lrest, K, y0


cartc, Lrest, K, y0 = returnparameters(la)
cartc2, Lrest2, _, y02 = returnparameters(la = la2, corneroffset = corneroffset)

cartcR2, LrestR2, _, y0R2 = returnparameters(la = la2, corneroffset=0.0707, ringoffset=ringoffset) # TODO: Number a rough estimate adapted from SMAP code. Research needed. 
cartc2R2, Lrest2R2, _, y02R2 = returnparameters(la = la, corneroffset = 0.2776, ringoffset=ringoffset) # TODO: Number a rough estimate adapted from SMAP code. Research needed. 

randf, randf2 = ForcesMultivariateNorm(cartc, cartc2, symmetry, magnitude = magnitude)   #TODO: refactor   

start = timeit.default_timer()

# Solve ODE
sol = solve_ivp(NPC, [0,40], y0, method='RK45', args=(Lrest, la, K, ka, randf, d, n, symmetry))
sol2 = solve_ivp(NPC, [0,40], y02, method='RK45', args=(Lrest2, la2, K, ka, randf2, d, n, symmetry))
#2nd ring
solR2 = solve_ivp(NPC, [0,40], y0R2, method='RK45', args=(LrestR2, la2, K, ka, randf, d, n, symmetry))
sol2R2 = solve_ivp(NPC, [0,40], y02R2, method='RK45', args=(Lrest2R2, la, K, ka, randf2, d, n, symmetry))

stop = timeit.default_timer()
print('Time: ', stop - start) 

#### Plotting ####################################################################
plt.rcParams.update({'font.size': 30})

def Plotting(sol, symmetry = symmetry, n = n,  linestyle = "-", legend = False, colourcode = True, colourbar = True, mainmarkercolor = "black"): # TODO 
    '''
    sol: Output of solve_ivp
    symmetry: number of nodes
    n: number of neighbours connected on each side per node
    linestyle (default: "-"): Linestyle in 1st plot 
    legend (default: False): Whether to show a legend in the 1st plot 
    colourcode (default: True): colourcodes trajectories in 2nd plot if True
    colorubar (default: True): Plots colourbar in 2nd plot if True and if colourcode is True
    mainmarkercolor: Colour of nodes in 2nd plot 
    '''
    t = sol.t # timepoints
    solplot = sol.y.T # positions and velocities of nodes over time
    solplot2D = np.reshape(solplot,(len(solplot),2*symmetry,2)) # 2D array of positions and velocities over time 
    
    # Position over time
    label = ["x", "y"]
    palette = sns.color_palette("hsv", 2*symmetry)
    for i in range(2*symmetry):
        axs[0].plot(t, solplot[:, i], label = label[i%2] + str(i), linestyle = linestyle, color = palette[i])
    if(legend):
        axs[0].legend(loc = 'best')
    axs[0].set(xlabel = 't (a.u.)')
    
    # Nodes at last timestep
    axs[1].plot(solplot2D[-1, :symmetry, 0], solplot2D[-1,:symmetry,1], 
    linestyle = "", marker = "o", color="gray", markerfacecolor = mainmarkercolor, markersize = 25, zorder = 50)
    
    # Anchor springs
    axs[1].plot([0,0], [0,0], marker = "o", color = "lightgray", markersize = 15)
    for i in range(0, symmetry):
        axs[1].plot((solplot2D[-1,i,0], 0), (solplot2D[-1,i,1], 0),
        linestyle = ":", marker = "", color="lightgray")   
        
    # Radial springs 
    for ni in range(1, n+1): # neighbours to connect to
        for i in range(symmetry): # node to connect from 
            axs[1].plot(solplot2D[-1, (i, (i+ni)%symmetry), 0], solplot2D[-1, (i, (i+ni)%symmetry), 1], 
            linestyle = ":", marker = "", color="gray")

    # Trajectories 
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
               
            lc = LineCollection(segments, cmap = 'plasma', norm=norm, zorder = 100)
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
            axs.plot(solplot2D[:,i,0], solplot[:,i,1], color = "blue", linestyle = "-")
        
        axs.axis("scaled")    
        axs.set(xlabel = "x (a.u.)", ylabel = "y (a.u.)")    
        plt.tight_layout()                  
   

fig, axs = plt.subplots(2, 1, figsize = (15, 26))
Plotting(sol, legend = True)
Plotting(sol2, linestyle="--", colourbar = False, mainmarkercolor="darkblue")
# Plotting(solR2, colourbar = False)
# Plotting(sol2R2, linestyle="--", colourbar = False, mainmarkercolor="darkblue")


## 3D plot
solplot2D0 = np.reshape(sol.y.T,(len(sol.y.T),2*symmetry,2)) # 2D array of positions and velocities over time 
solplot2D1 = np.reshape(sol2.y.T,(len(sol2.y.T),2*symmetry,2))
solplot2D2 = np.reshape(solR2.y.T,(len(solR2.y.T),2*symmetry,2))
solplot2D3 = np.reshape(sol2R2.y.T,(len(sol2R2.y.T),2*symmetry,2))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(solplot2D0[-1, :symmetry,0], solplot2D0[-1, :symmetry,1])
ax.scatter(solplot2D1[-1, :symmetry,0], solplot2D1[-1, :symmetry,1])
ax.scatter(solplot2D2[-1, :symmetry,0], solplot2D2[-1, :symmetry,1], -50)
ax.scatter(solplot2D3[-1, :symmetry,0], solplot2D3[-1, :symmetry,1], -50)




#plt.scatter(F[:,0], F[:,1]) 
#cartc = np.array([1.,0,np.sqrt(2)/2, np.sqrt(2)/2,0,1,-np.sqrt(2)/2,np.sqrt(2)/2,-1,0,-np.sqrt(2)/2,-np.sqrt(2)/2,0,-1,np.sqrt(2)/2,-np.sqrt(2)/2]) #TODO remove


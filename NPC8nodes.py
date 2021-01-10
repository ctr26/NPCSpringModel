# -*- coding: utf-8 -*-
"""

"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def octagonspring(y, t):
    # external force 
    fext = np.array([[0.5,0.],[0.,0.],[0.,0.],[0.,0.],[0.5,0.],[0.,0.],[0.,0.],[0.,0.]])
    d = 0.9
    n = 3 # maximum distant neighbour to connect on each side 
    k1 = k2 = k3 = 1.
    k4 = 0.5
    ka = 1.
    
    x0x, x0y, x1x, x1y, x2x, x2y, x3x, x3y, x4x, x4y, x5x, x5y, x6x, x6y, x7x, x7y, v0x, v0y, v1x, v1y, v2x, v2y, v3x, v3y, v4x, v4y, v5x, v5y, v6x, v6y, v7x, v7y = y
    
    # All lengths in a fully connected octagon with r = 1 and central anchor 
    l1 = np.sqrt(2-2*np.cos(0.25*np.pi))
    l2 = np.sqrt(2)
    l3 = np.sqrt(2-2*np.cos(0.75*np.pi))
    l4 = 2.
    la = 1.
    x = np.array([[x0x, x0y], [x1x, x1y], [x2x, x2y], [x3x, x3y], [x4x, x4y], [x5x, x5y], [x6x, x6y], [x7x, x7y]])
    v = np.array([[v0x, v0y], [v1x, v1y], [v2x, v2y], [v3x, v3y], [v4x, v4y], [v5x, v5y], [v6x, v6y], [v7x, v7y]])
    a = np.array([0., 0.])
    
    Lrest = np.array(
        [[0., l1, l2, l3, l4, l3, l2, l1],
         [l1, 0., l1, l2, l3, l4, l3, l2],
         [l2, l1, 0., l1, l2, l3, l4, l3],
         [l3, l2, l1, 0., l1, l2, l3, l4],
         [l4, l3, l2, l1, 0., l1, l2, l3],
         [l3, l4, l3, l2, l1, 0., l1, l2], 
         [l2, l3, l4, l3, l2, l1, 0., l1],
         [l1, l2, l3, l4, l3, l2, l1, 0.]])
            
    K = np.array(
        [[0., k1, k2, k3, k4, k3, k2, k1],
         [k1, 0., k1, k2, k3, k4, k3, k2],
         [k2, k1, 0., k1, k2, k3, k4, k3],
         [k3, k2, k1, 0., k1, k2, k3, k4],
         [k4, k3, k2, k1, 0., k1, k2, k3],
         [k3, k4, k3, k2, k1, 0., k1, k2], 
         [k2, k3, k4, k3, k2, k1, 0., k1],
         [k1, k2, k3, k4, k3, k2, k1, 0.]])
     
    allaccarray = np.zeros((8,2)) # array for accelerations of node 0 - 7 
    
    for i in range(8):
        
        accarray = np.array([0., 0.]) # initiate acceleration array for each node i 
        
        for j in [k for k in range(-n, n+1) if k != 0]: # - n to + n, skipping 0 
            
            jnew = (i+j)%8
            accarray += K[i][jnew] * (x[i]-x[jnew])/np.linalg.norm(x[i]-x[jnew]) * (Lrest[i][jnew] - np.linalg.norm(x[i]-x[jnew]))

        accarray += ka * (x[i] - a)/np.linalg.norm(x[i] - a) * (la - np.linalg.norm(x[i] - a)) # anchor
        accarray = fext[i] + accarray - d*v[i] # external force and damping
        allaccarray[i] = accarray 

    dxdt = np.concatenate((v.flatten(), allaccarray.flatten()))  
                                                                
    return dxdt
    
    
    # initial conditions
    
def pol2cart(rho, phi):
    '''Function transforms polar to cartesian coordinates'''
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

angle = 0.
cartc = np.zeros(16)
for i in range(0,16,2): # skip every other value to populate i+1 with y 
    x, y = pol2cart(1,angle)
    cartc[i] = x
    cartc[i+1] = y
    angle += 0.25*np.pi
    
y0 = np.concatenate((cartc, np.zeros(16))) # last 16 entries are starting velocities 
t = np.linspace(0, 25, 100)

sol8 = odeint(octagonspring, y0, t) 
solplot = sol8

#### Plotting
fig, axs = plt.subplots(1, 2 ,figsize = (18, 13))
axs = axs.ravel()

# x and y position over time 
label = ["x0x", "x0y", "x1x", "x1y", "x2x", "x2y", "x3x", "x3y", "x4x", "x4y", "x5x", "x5y", "x6x", "x6y", "x7x", "x7y"]
for i in range(16):
    axs[0].plot(t, solplot[:, i], label = label[i])
axs[0].set(xlabel = 't')
axs[0].legend(loc = 'best')


    
axs[1].plot(solplot[-1,(0,2,4,6,8,10,12,14,0)], solplot[-1,(1,3,5,7,9,11,13,15,1)], linestyle = "-", marker = "o", color="white", markerfacecolor = "black", markersize = 10)

#trajectory
for i in range(0,16,2):
    axs[1].plot(solplot[:,i], solplot[:,i+1], color = "blue", linestyle = "-")

axs[1].axis("scaled")

axs[1].set(xlabel = "x", ylabel = "y")

plt.tight_layout()
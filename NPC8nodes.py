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
import csv
import matplotlib.animation as animation
from IPython.display import HTML
from matplotlib import rc
from matplotlib.patches import FancyArrowPatch

### Parameters
symmetry = 8
d = 1
n = 3 
magnitude = 10
ka = 1
la = 50
corneroffset0 = 0
ringoffset = 0

solution = DeformNPC(symmetry, d, n, magnitude, ka, la, corneroffset0, ringoffset).sol

class DeformNPC:
#    symmetry = 8 # finegraining with multiples of 8 possible 
#    d = 1#0.1 # damping 
#    n = 2 # n maximum distant neighbour to connect on each side 
#    magnitude = 25 # Average magnitude of distortion per ring   
    

        

    #rings = 1 # Number of rings TODO: Doesn't do anything currently
#    ka = 0.5 # Spring constant anchor springs
    
    ### NPC Measures. Here rough measures for Nup107, adapted from SMAP. TODO: Research measures. 
#    la = 50 # TODO: Number a rough estimate adapted from SMAP code. Research needed. 
#    la2 = 54 # TODO: Number a rough estimate adapted from SMAP code. Research needed. 
    
#    zdist = -50 # distance between cytoplasmic and nucleoplasmic ring TODO: realistic number. move outside of class?
    
    # Ring 1
#    corneroffset0 = 0
#    corneroffset1 = 0.2069 + np.random.normal(0,0) # TODO: Number a rough estimate adapted from SMAP code. Research needed. 
    
    # Ring 2 
#    ringoffset = 0.17#0.1309 + np.random.normal(0,0) # Offset between nucleoplamic and cytoplasmic ring. TODO: Number a rough estimate adapted from SMAP code. Research needed. 
#    corneroffset2 = 0.0707 + np.random.normal(0,0) # TODO: Number a rough estimate adapted from SMAP code. Research needed. 
#    corneroffset3 = 0.2776 + np.random.normal(0,0) # TODO: Number a rough estimate adapted from SMAP code. Research needed. 
    ### 

    def __init__(self, symmetry, d, n, magnitude, ka, la, corneroffset0, ringoffset):
        self.symmetry = symmetry
        self.d = d
        self.n = n
        self.magnitude = magnitude
        self.ka = ka
        self.la = la
        self.corneroffset0 = corneroffset0
        self.ringoffset = ringoffset

    
        if(n > symmetry/2):
            self.n = int(np.floor(symmetry/2))
            warn("Selected number of neighbours n too large. n has been set to " + str(n) + ".")
        cartc, Lrest, K, y0 = self.returnparameters(la, corneroffset = 0, ringoffset = 0, symmetry = self.symmetry)
        randf = self.ForcesMultivariateNorm(cartc)
        
        tlast = 40
        tspan = [0,tlast]
    #teval = np.arange(0,tlast,0.2)
        teval = None
    
        # Solve ODE, ring 1 - 4 
        self.sol = solve_ivp(self.NPC, tspan, y0, t_eval=teval, method='RK45', args=(Lrest, la, K, ka, randf, d, n, symmetry))
            
    ### Functions 
    
    def NPC(self,t, y, Lrest, la, K, ka, randf, d, n, symmetry = 8):
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
    
    
    def Pol2cart(self, rho, phi):
        '''Transforms polar coordinates of a point (rho: radius, phi: angle) to 2D cartesian coordinates.
        '''
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return(x, y)
    
    def Initialcoords(self, la, symmetry = 8, corneroffset = 0, ringoffset = 0): #TODO real-data coordinates
        '''
        Generates cartesian coordinates of the NPC given radius and symmetry 
        ## Input ##
        la: NPC Radius
        symmetry: Number of corners
        angleoffset (rad): Rotates the NPC by the given offset. Default 0
        ## Return values ##
        Cartesian coordinates in 1D and 2D array format 
        '''
        forces = np.zeros(symmetry) 
        angle = 0.
        cartc = np.zeros(2*symmetry) 
        
        for i in range(0, 2*symmetry, 2): # skip every other entry to populate it with y-coords
            x, y = self.Pol2cart(la + forces[int(i/2)], angle+corneroffset+ringoffset)
            cartc[i] = x
            cartc[i+1] = y
            angle += 2*np.pi/symmetry
        return cartc
    
    
    def Springlengths(self, cartc, symmetry): 
        '''Compute lengths of springs from coordinates and returns circulant matrix
        '''
        cart2D = cartc.reshape(symmetry,2)
        l = np.zeros(symmetry)
        for i in range(len(l)):
            l[i] = np.linalg.norm(cart2D[0, :] - cart2D[i, :])      
        return circulant(l)
    
    
    def Springconstants(self, symmetry): # TODO: More variable spring constants? 
        "Returns circulant matrix of spring constants "
        k = np.ones(int(np.floor(symmetry/2)))
        if(symmetry%2 == 0): #if symmetry is even
            k[-1] = k[-1]/2 # springs that connect opposite corners will be double-counted. Spring constant thus halved 
            K = circulant(np.append(0, np.append(k, np.flip(k[:-1]))))
        else: #if symmetry is odd 
            K = circulant(np.append(0, [k, np.flip(k)]))
        return K
    
    
    def ForcesMultivariateNorm(self, *allringcords, symmetry = 8, magnitude = 50): # TODO: include distances to nucleoplasmic ring 
        '''
        Returns array of Forces that are later applied in radial direction to the NPC corners
        ## Input ## 
        *coordring: Initial coordinates of nodes for an arbitrary number of rings. 
        magnitude: Total magnitude of distortion. Default is 50. 
        ## Returns ## 
        For each ring, an array of forces applied to each node
        '''
        #allcoords = np.asarray([cartc, cartc2, cartcR2, cartc2R2])#TODO
        allcoords = np.asarray(allringcords) 
        nrings = len(allringcords) # number of rings
        #nrings = 4 # TODO
        allcoords = allcoords.reshape(symmetry*nrings, 2)
      
        AllD = np.zeros((symmetry*nrings, symmetry*nrings)) # all distances
        
        for i in range(symmetry*nrings):
            for j in range(symmetry*nrings):
                AllD[i, j] = np.linalg.norm(allcoords[i, :] - allcoords[j, :])
    
        mean = list(np.zeros(symmetry*nrings)) # Mean of the normal distribution
    
        LInv = AllD.max() - AllD # invert distances  
        
        global cov
        cov = [] # covariance matrix 
        for i in range(symmetry*nrings):
            cov.append(list(LInv[i]/AllD.max()))
                
        rng = np.random.default_rng(seed = 2) # TODO remove seed
        
        global F
        F = rng.multivariate_normal(mean, cov)#, size = 1000) # TODO
        
    
        initialmag = sum(abs(F))
        
        if (initialmag != 0):
            F = nrings*magnitude/initialmag * F
        
        return np.split(F, nrings)#np.array_split(F, nrings)
      
    
    def returnparameters(self, la, corneroffset, ringoffset, symmetry):
        cartc = self.Initialcoords(la = la, symmetry=symmetry, corneroffset=corneroffset, ringoffset=ringoffset)
        Lrest = self.Springlengths(cartc, symmetry)
        K = self.Springconstants(symmetry)
        y0 = np.concatenate((cartc, np.zeros(2*symmetry))) # starting coordinates and velocities of nodes. last half of the entries are starting velocities 
        
        return cartc, Lrest, K, y0
    
    
    #cartc, Lrest, K, y0 = returnparameters(la, corneroffset = corneroffset0)
    # cartc2, Lrest2, _, y02 = returnparameters(la = la2, corneroffset = corneroffset1)
    # cartcR2, LrestR2, _, y0R2 = returnparameters(la = la2, corneroffset=corneroffset2, ringoffset=ringoffset) # TODO: Number a rough estimate adapted from SMAP code. Research needed. 
    # cartc2R2, Lrest2R2, _, y02R2 = returnparameters(la = la, corneroffset = corneroffset3, ringoffset=ringoffset) # TODO: Number a rough estimate adapted from SMAP code. Research needed. 
    
    #randf = ForcesMultivariateNorm(cartc)
    #randf, randf2, randf3, randf4 = ForcesMultivariateNorm(cartc, cartc2, cartcR2, cartc2R2, symmetry = symmetry, magnitude = magnitude)   

# force coordinates    
    # fcoords = Initialcoords(la, forces = randf, corneroffset = corneroffset0)
    # fcoords2 = Initialcoords(la2, forces = randf2, corneroffset = corneroffset1)
    # fcoords3 = Initialcoords(la2, forces = randf3, corneroffset=corneroffset2, ringoffset=ringoffset)
    # fcoords4 = Initialcoords(la, forces = randf4, corneroffset = corneroffset3, ringoffset=ringoffset)
    
    #tlast = 40
    #tspan = [0,tlast]
    #teval = np.arange(0,tlast,0.2)
    #teval = None
    
    # Solve ODE, ring 1 - 4 
    #sol = solve_ivp(NPC, tspan, y0, t_eval=teval, method='RK45', args=(Lrest, la, K, ka, randf, d, n, symmetry))
    # sol2 = solve_ivp(NPC, tspan, y02, t_eval=teval, method='RK45', args=(Lrest2, la2, K, ka, randf2, d, n, symmetry))
    # solR2 = solve_ivp(NPC, tspan, y0R2, t_eval=teval, method='RK45', args=(LrestR2, la2, K, ka, randf3, d, n, symmetry))
    # sol2R2 = solve_ivp(NPC, tspan, y02R2, t_eval=teval, method='RK45', args=(Lrest2R2, la, K, ka, randf4, d, n, symmetry))
    






























#### Plotting ####################################################################
plt.rcParams.update({'font.size': 50})

def Plotting(sol, symmetry = symmetry, n = n,  linestyle = "-", legend = False, trajectory = True, colourcode = True, colourbar = True, mainmarkercolor = "black", markersize = 25, forces = 0, showforce = False): # TODO 
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
    frame = -1 # 0 is the first frame, -1 is the last frame
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
    axs[1].plot(solplot2D[frame, :symmetry, 0], solplot2D[frame,:symmetry,1], 
    linestyle = "", marker = "o", color="gray", markerfacecolor = mainmarkercolor, markersize = markersize, zorder = 50)
    
    # Anchor springs
    axs[1].plot([0,0], [0,0], marker = "o", color = "lightgray", markersize = 15)
    for i in range(0, symmetry):
        axs[1].plot((solplot2D[frame,i,0], 0), (solplot2D[frame,i,1], 0),
        linestyle = ":", marker = "", color="lightgray")   
        
    # Radial springs 
    for ni in range(1, n+1): # neighbours to connect to
        for i in range(symmetry): # node to connect from 
            axs[1].plot(solplot2D[frame, (i, (i+ni)%symmetry), 0], solplot2D[frame, (i, (i+ni)%symmetry), 1], 
            linestyle = ":", marker = "", color="gray")#, linewidth = 5)

    # Trajectories 
    if (trajectory):
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
            
        else: # monochrome trajectory
            for i in range(0,symmetry,2):
                axs[1].plot(solplot2D[:,i,0], solplot2D[:,i,1], color = "blue", linestyle = "-")

    ### Force vectors
    if(showforce and type(forces) != int):
        forces2d = forces.reshape(symmetry, 2)
        for i in range(0, symmetry):
            axs[1].arrow(x = solplot2D[0,i,0], y = solplot2D[0,i,1], 
                         dx = (forces2d[i,0] - solplot2D[0,i,0]), dy = (forces2d[i,1] - solplot2D[0,i,1]),
                         width = 0.7, color="blue")   
            
    axs[1].axis("scaled")
    axs[1].set(xlabel = "x (nm)", ylabel = "y (nm)")  
    plt.tight_layout()
             

def Plotforces(forces, coords):  
    fig, ax1 = plt.subplots(1,1, figsize = (10, 10))
    nodes = int(len(forces)/2)
    forces2d = forces.reshape(nodes, 2)
    start = coords.reshape(nodes, 2)
    for i in range(int(nodes/2)):
        ax1.arrow(x = start[i,0], y = start[i,1], 
        dx = (forces2d[i,0] - start[i,0]), dy = (forces2d[i,1] - start[i,1]),
        width = 0.7, color="blue") 
    plt.axis("scaled")


Plotforces(np.concatenate([fcoords, fcoords2, fcoords3, fcoords4]), np.concatenate([cartc, cartc2, cartcR2, cartc2R2]))


showforces = False
trajectories = False
fig, axs = plt.subplots(2, 1, figsize = (15, 26))
Plotting(solR2, colourbar = False, mainmarkercolor="darkgray", legend = False, forces = fcoords3, showforce = showforces, trajectory=trajectories)#, markersize = 30)
Plotting(sol2R2, linestyle="--", colourbar = False, mainmarkercolor="darkgray", forces = fcoords4, showforce = showforces, trajectory=trajectories)
Plotting(sol,  forces = fcoords, showforce= showforces, trajectory = trajectories)
Plotting(sol2, linestyle="--", colourbar = False, forces = fcoords2, showforce = showforces, trajectory = trajectories)


solplot2D0 = np.reshape(sol.y.T,(len(sol.y.T),2*symmetry,2)) # 2D array of positions and velocities over time 
solplot2D1 = np.reshape(sol2.y.T,(len(sol2.y.T),2*symmetry,2))
solplot2D2 = np.reshape(solR2.y.T,(len(solR2.y.T),2*symmetry,2))
solplot2D3 = np.reshape(sol2R2.y.T,(len(sol2R2.y.T),2*symmetry,2))

## 3D plot
frame = 0 # 0 is the first frame, -1 is the last frame
fig = plt.figure(figsize = (15,15))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(solplot2D0[frame, :symmetry,0], solplot2D0[frame, :symmetry,1], s = 200, c = "black")
ax.scatter(solplot2D1[frame, :symmetry,0], solplot2D1[frame, :symmetry,1], s = 200, c = "black")
ax.scatter(solplot2D2[frame, :symmetry,0], solplot2D2[frame, :symmetry,1], zdist, s = 200, c = "gray")
ax.scatter(solplot2D3[frame, :symmetry,0], solplot2D3[frame, :symmetry,1], zdist, s = 200, c = "gray")

# 3D arrows
forces2d = fcoords.reshape(symmetry, 2)
forces2d2 = fcoords2.reshape(symmetry, 2)
forces2d3 = fcoords3.reshape(symmetry, 2)
forces2d4 = fcoords4.reshape(symmetry, 2)

linewidth = 3
normalize = True
for i in range(0, symmetry):
    ax.quiver(solplot2D0[0,i,0], solplot2D0[0,i,1], 0, forces2d[i,0], forces2d[i,1], 0, length = randf[i], normalize = normalize, linewidth = linewidth , edgecolor = "blue")#(forces2d[i,0] - solplot2D0[0,i,0]),(forces2d[i,1] - solplot2D0[0,i,1]), 0)   
    ax.quiver(solplot2D1[0,i,0], solplot2D1[0,i,1], 0, forces2d2[i,0], forces2d2[i,1], 0, length = randf2[i], normalize = normalize, linewidth = linewidth, edgecolor = "blue")#(forces2d[i,0] - solplot2D0[0,i,0]),(forces2d[i,1] - solplot2D0[0,i,1]), 0)   
    ax.quiver(solplot2D2[0,i,0], solplot2D2[0,i,1], zdist, forces2d3[i,0], forces2d3[i,1], 0, length = randf3[i], normalize = normalize, linewidth = linewidth, edgecolor = "blue")#(forces2d[i,0] - solplot2D0[0,i,0]),(forces2d[i,1] - solplot2D0[0,i,1]), 0)   
    ax.quiver(solplot2D3[0,i,0], solplot2D3[0,i,1], zdist, forces2d4[i,0], forces2d4[i,1], 0, length = randf4[i], normalize = normalize, linewidth = linewidth, edgecolor = "blue")#(forces2d[i,0] - solplot2D0[0,i,0]),(forces2d[i,1] - solplot2D0[0,i,1]), 0)   



ax.set_xlabel('x (nm)', labelpad = 30)
ax.set_ylabel('y (nm)', labelpad = 30)
ax.set_zlabel('z (nm)', labelpad = 40)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

# Now set color to white (or whatever is "invisible")
ax.xaxis.pane.set_edgecolor('w')
ax.yaxis.pane.set_edgecolor('w')
ax.zaxis.pane.set_edgecolor('w')

# Bonus: To get rid of the grid as well:
ax.grid(False)
plt.show()
    



with open('/home/maria/Documents/NPCPython/NPCexampleposter.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for i in range(symmetry):
        spamwriter.writerow(np.append(solplot2D0[-1, i, :], 0))
    for i in range(symmetry):
        spamwriter.writerow(np.append(solplot2D1[-1, i, :], 0))
    for i in range(symmetry):
        spamwriter.writerow(np.append(solplot2D2[-1, i, :], zdist))
    for i in range(symmetry):
        spamwriter.writerow(np.append(solplot2D3[-1, i, :], zdist))




#plt.scatter(F[:,0], F[:,1]) 
#cartc = np.array([1.,0,np.sqrt(2)/2, np.sqrt(2)/2,0,1,-np.sqrt(2)/2,np.sqrt(2)/2,-1,0,-np.sqrt(2)/2,-np.sqrt(2)/2,0,-1,np.sqrt(2)/2,-np.sqrt(2)/2]) #TODO remove
###### TODO
global xy
global xy1
global xy2
global xy3
global xyA
xy = solplot2D0[:, np.append(np.arange(symmetry), 0), :] # TODO 
xy1 = solplot2D1[:, np.append(np.arange(symmetry), 0), :] # TODO 
xy2 = solplot2D2[:, np.append(np.arange(symmetry), 0), :] # TODO 
xy3 = solplot2D3[:, np.append(np.arange(symmetry), 0), :] # TODO 

# xy = solplot2D0[:, :symmetry, :] # TODO 
# xy1 = solplot2D1[:, :symmetry, :] # TODO 
# xy2 = solplot2D2[:, :symmetry, :] # TODO 
# xy3 = solplot2D3[:, :symmetry, :] # TODO 

# xyA = np.zeros((len(xy),2*symmetry,2))
# for i in range(len(xy)):
#     xyA[i,:,0] = np.insert(xy[i,:symmetry,0], np.arange(symmetry), 0)
#     xyA[i,:,1] = np.insert(xy[i,:symmetry,1], np.arange(symmetry), 0)

#%matplotlib qt # Run in console if python doesn't show plot 
class AnimatedScatter(object):
    """An animated scatter plot using matplotlib.animations.FuncAnimation."""
    def __init__(self, xy, numpoints=symmetry+1):
        self.numpoints = numpoints
        self.stream = self.data_stream(xy)
        # Setup the figure and axes...
        self.fig, self.ax = plt.subplots(figsize = (9, 10))
        plt.rcParams.update({'font.size': 20})
        # Then setup FuncAnimation.
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=15, frames = len(xy),
                                          init_func=self.setup_plot, blit=True)
        #HTML(self.ani.to_html5_video())
        self.ani.save("Damping0.mp4", dpi = 250)

    def setup_plot(self):
        """Initial drawing of the scatter plot."""
#        x, y, x1, y1, x2, y2, x3, y3 = next(self.stream).T
        # 4 to plot x-y coords
        # 4 to plot anchor springs 
        # n * symmetry * rings to plot circumferential spring

        self.lines = []
        for index in range(int(8 + 8*2*4)):
            if (index <= 1):
                self.lobj = self.ax.plot([], [], marker = "o", color = "gray", markerfacecolor = "gray", linestyle = "-", markersize = 15)
            elif (index > 1 and index <=3):
                self.lobj = self.ax.plot([], [], marker = "o", color = "gray", markerfacecolor = "black", linestyle = "-", markersize = 15)
            elif (index > 3 and index <= 7):
                self.lobj = self.ax.plot([], [], marker = "", color = "lightgray", linestyle = "-")

            else:
                self.lobj = self.ax.plot([], [], marker = "", color = "gray", linestyle = "-")

            self.lines.append(self.lobj)
        
        self.ax.axis("scaled")
        self.ax.set(xlabel = "x (nm)", ylabel = "y (nm)")  
        self.ax.axis([-80, 80, -80, 80])
        return [self.lines[i][0] for i in range(int(8 + 8*2*4))]#self.lines[0][0], self.lines[1][0], self.lines[2][0], self.lines[3][0], self.lines[4][0], self.lines[5][0],self.lines[6][0], self.lines[7][0],#self.line, self.line1, self.line2, self.line3, self.lineA, self.lineA1, self.lineA2, self.lineA3,

    def data_stream(self, pos):
        while True:
            for i in range(0, len(xy)):
                x_y = xy[i]
                x_y1 = xy1[i]
                x_y2 = xy2[i]
                x_y3 = xy3[i]
                yield np.c_[x_y[:,0], x_y[:,1], x_y1[:,0], x_y1[:,1], x_y2[:,0], x_y2[:,1], x_y3[:,0], x_y3[:,1]]

    def update(self, i):
        """Update the plot."""
        x, y, x1, y1, x2, y2, x3, y3 = next(self.stream).T
        

        xa = np.insert(x[:symmetry], np.arange(symmetry), 0)
        ya = np.insert(y[:symmetry], np.arange(symmetry), 0)
        x1a = np.insert(x1[:symmetry], np.arange(symmetry), 0)
        y1a = np.insert(y1[:symmetry], np.arange(symmetry), 0)       
        x2a = np.insert(x2[:symmetry], np.arange(symmetry), 0)
        y2a = np.insert(y2[:symmetry], np.arange(symmetry), 0)        
        x3a = np.insert(x3[:symmetry], np.arange(symmetry), 0)
        y3a = np.insert(y3[:symmetry], np.arange(symmetry), 0)
        


        xlist = [x, x1, x2, x3, xa, x1a, x2a, x3a]
        ylist = [y, y1, y2, y3, ya, y1a, y2a, y3a]

                
        for lnum, self.line in enumerate(self.lines):
            if lnum >= len(xlist):
                break
            self.line[0].set_data(xlist[lnum], ylist[lnum]) 

        n = 2 # TODO 
        count = len(xlist)
        for lnum in range(4):
            for ni in range(1, n+1): # neighbours to connect to
                for i in range(symmetry): # node to connect from 
                    self.lines[count][0].set_data((xlist[lnum][i], xlist[lnum][(i+ni)%symmetry]), (ylist[lnum][i], ylist[lnum][(i+ni)%symmetry]))
                    count += 1
        
        return [self.lines[i][0] for i in range(int(8 + 8*2*4))]#self.lines[0][0], self.lines[1][0], self.lines[2][0], self.lines[3][0], self.lines[4][0], self.lines[5][0],self.lines[6][0], self.lines[7][0], #self.line, self.line1, self.line2, self.line3, self.lineA, self.lineA1, self.lineA2, self.lineA3,


if __name__ == '__main__':
    #a = AnimatedScatter(xy)

    plt.show()




# x = np.random.random(3)
# y = np.random.random(3)
# x1 = np.random.random(3)
# y1 = np.random.random(3)
# fig, ax = plt.subplots()

# ax.axis([-1, 1, -1, 1])

# lines = []
# for index in range(2):
#     lobj = ax.plot([], [], marker = "o", color = "black", linestyle = ":")
#     lines.append(lobj)

# # for line in lines:
# #     line[0].set_data([],[])
    
# xlist = [x, x1]
# ylist = [y, y1]
# for lnum, line in enumerate(lines):
#     line[0].set_data(xlist[lnum],ylist[lnum]) 
# plt.show()

# for i in range(10):
#    if(i >5):
#        break
#    print(i)
   
# plt.rcParams.update({'font.size': 2})
# fig, axs = plt.subplots(32, 32, figsize = (16,16))
# for i in range(32):
#    for j in range(32):
#        axs[i,j].scatter(F[:,i], F[:,j], s = 4)
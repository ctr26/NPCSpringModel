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
symmet = 8      # Rotational symmetry of the NPC
mag = 100        # Magnitude of deformation 
nConnect = 3    # Number of connected neighbour nodes in clock-wise and anti-clockwise direction
nRings = 4      # Number of rings 

class DeformNPC:
    def __init__(self, symmet, nConnect, mag, nRings = 1, r = 0, ringAngles = 0):
        self.symmet = symmet
        self.initcoords = []        
        self.solution = []
        self.fcoords = []    
        
        damp = 1 # damping
        kr = 0.7 # spring constant of anchor spring 
        
        tlast = 40
        tspan = [0,tlast]      
        teval = np.arange(0,tlast,0.4)
        #teval = None        
        

        Lrests = []
        Ks = []
        y0s = [] 

        if(nConnect > self.symmet/2):
            nConnect = int(np.floor(self.symmet/2))
            warn("Selected number of neighbours nConnect too large. nConnect has been set to " + str(nConnect) + ".")
        
        if(len(r) != nRings or len(ringAngles) != nRings):
            warn("r and ringAngles must be the same length as nRings")


        global initcoord
        for i in range(nRings):            
            initcoord = self.Initialcoords(r[i], ringAngle = ringAngles[i])
            self.initcoords.append(initcoord) # TODO remove .flatten()
                       
            Lrests.append(self.Springlengths(initcoord)) 
            Ks.append(self.Springconstants())
            y0s.append(np.concatenate((initcoord.flatten(), np.zeros(2 * self.symmet)))) #

        self.randfs = self.ForcesMultivariateNorm(self.initcoords, mag, nRings = nRings) # TODO
            
        # Solve ODE, ring 1 - 4 

        
        for i in range(nRings):
            self.solution.append(solve_ivp(self.NPC, tspan, y0s[i], t_eval=teval, method='RK45', args=(Lrests[i], r[i], Ks[i], kr, self.randfs[i], damp, nConnect)))    
            self.fcoords.append(self.Initialcoords(r[i], ringAngles[i], self.randfs[i]))
            
            
    ### Methods 
    
    def NPC(self, t, y, Lrest, r, K, kr, randfs, damp, nConnect):
        '''
        t: time points 
        y: values of the solution at t 
        Lrest: Circulant matrix of resting lengths of all springs 
        K: Circulant matrix of all radial spring constants 
        kr: Spring constants of anchor springs 
        randfs: array of forces (length = symmet) to be applied in radial direction to each node 
        d: Damping factor 
        nConnect: Number of connected neighbours in cw and ccw direction for each node 
        symmet (default: 8): Number of nodes 
        output: solutions at t. x and y components of positions and velocities of each node for each time-step 
        '''
        v = np.reshape(y[2*self.symmet:], (self.symmet, 2))
        x = np.reshape(y[:2*self.symmet], (self.symmet, 2))
        
        anc = np.array([0., 0.]) # coordinates of anchor node   
        F = np.zeros((self.symmet, 2)) # Forces
        
        for i in range(self.symmet): # TODO test
            F[i] = randfs[i]*x[i] / np.linalg.norm([x[i], anc]) 
    
        allaccarray = np.zeros((self.symmet, 2)) # array for accelerations of node 0 - 7
        
        for i in range(self.symmet): # i indicates the reference node        
            accarray = np.array([0., 0.]) # initiate acceleration array for each node i 
            
            for j in [k for k in range(-nConnect, nConnect+1) if k != 0]: # j is neighbour nodes -nConnect to +nConnect relative to i, skipping 0 (0=i)            
                jnew = (i+j)%self.symmet 
                accarray += K[i][jnew]  * (x[i]-x[jnew])/np.linalg.norm(x[i]-x[jnew]) * (Lrest[i][jnew] - np.linalg.norm(x[i]-x[jnew]))
    
            accarray += kr * (x[i] - anc)/np.linalg.norm(x[i] - anc) * (r - np.linalg.norm(x[i] - anc)) #anchor
            accarray = F[i] + accarray - damp*v[i]  # external force and damping
            allaccarray[i] = accarray 
    
        dxdt = np.concatenate((v.flatten(), allaccarray.flatten()))                                                                
        return dxdt
    
    
    def Pol2cart(self, rho, phi):
        '''Transforms polar coordinates of a point (rho: radius, phi: angle) to 2D cartesian coordinates.
        '''
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return(x, y)
    
    def Initialcoords(self, r, ringAngle = 0, forces = 0): #TODO real-data coordinates
        '''
        Generates cartesian coordinates of the NPC given radius and self.symmet 
        ## Input ##
        r: NPC Radius

        angleoffset (rad): Rotates the NPC by the given offset. Default 0
        ## Return values ##
        Cartesian coordinates in 1D and 2D array format 
        '''
        if(type(forces) != int):
            if (len(forces) != self.symmet):
                warn("forces must be 0 or an array with len(self.symmet")
                
        forces = forces * np.ones(self.symmet) # forces is 0 or np.array with len(self.symmet)
        rotAngle = 0.

        initcoord = np.zeros((self.symmet, 2))
        
        for i in range(self.symmet):
            initcoord[i, 0], initcoord[i, 1] = self.Pol2cart(r + forces[i], 
                                                             rotAngle+ringAngle)
            rotAngle += 2*np.pi/self.symmet

        return initcoord
    
    
    def Springlengths(self, initcoord): 
        '''Compute lengths of springs from coordinates and returns circulant matrix
        '''
        #cart2D = initcoord.reshape(self.symmet,2)
        l = np.zeros(self.symmet)
        for i in range(self.symmet):
            l[i] = np.linalg.norm(initcoord[0, :] - initcoord[i, :])      
        return circulant(l)
    
    
    def Springconstants(self): # TODO: More variable spring constants? 
        "Returns circulant matrix of spring constants "
        k = np.ones(int(np.floor(self.symmet/2)))
        if(self.symmet%2 == 0): #if symmet is even
            k[-1] = k[-1]/2 # springs that connect opposite corners will be double-counted. Spring constant thus halved 
            K = circulant(np.append(0, np.append(k, np.flip(k[:-1]))))
        else: #if symmet is odd 
            K = circulant(np.append(0, [k, np.flip(k)]))
        return K
    
    
    
    def ForcesMultivariateNorm(self, initcoords, mag, nRings = 1): # TODO: include distances to nucleoplasmic ring 
        '''
        Returns array of Forces that are later applied in radial direction to the NPC corners
        ## Input ## 
        *coordring: Initial coordinates of nodes for an arbitrary number of rings. 
        mag: Total magnitude of distortion. Default is 50. 
        ## Returns ## 
        For each ring, an array of forces applied to each node
        '''
        allcoords = np.asarray(initcoords) 
        allcoords = allcoords.reshape(self.symmet*nRings, 2)
      
        AllD = np.zeros((self.symmet*nRings, self.symmet*nRings)) # all distances
        
        for i in range(self.symmet*nRings):
            for j in range(self.symmet*nRings):
                AllD[i, j] = np.linalg.norm(allcoords[i, :] - allcoords[j, :])
    
        mean = list(np.zeros(self.symmet*nRings)) # Mean of the normal distribution
    
        LInv = AllD.max() - AllD # invert distances  

        cov = [] # covariance matrix 
        for i in range(self.symmet*nRings):
            cov.append(list(LInv[i]/AllD.max()))
                
        rng = np.random.default_rng() # TODO remove seed
        
        #global F
        F = rng.multivariate_normal(mean, cov)#, size = 1000) # TODO
        
    
        initmag = sum(abs(F))
        
        if (initmag != 0):
            F = nRings*mag/initmag * F
        
        if (nRings) == 1: # TODO: more general return statement 
            return np.split(F, nRings)[0]
        else:
            return np.split(F, nRings)#np.array_split(F, nrings)


deformNPC = DeformNPC(symmet, nConnect, mag, nRings = nRings, r = [50, 54, 54, 50], ringAngles = [0, 0.2069, 0.0707, 0.2776])

solution = deformNPC.solution
fcoords = deformNPC.fcoords # coordinates of force vectors
initcoords = deformNPC.initcoords # starting coordinates 
randfs = deformNPC.randfs # magnitude of force vectors 

def Sol2D(solution):
    """"""
    nFrames = len(solution.t)
    symmet = int(len(solution.y)/4) #/4 because of 2 dim times both velocity and position 
    pos1D = solution.y.T[: , :2*symmet] # positions over time
    vel1D = solution.y.T[: , 2*symmet:] # velocities over time
    pos2D = np.reshape(pos1D, (nFrames, symmet, 2))
    vel2D = np.reshape(vel1D, (nFrames, symmet, 2))
    return pos2D, vel2D



def Pos2D(solution):
    pos2D, vel2D = Sol2D(solution)
    return pos2D   

#### Plotting ####################################################################
plt.rcParams.update({'font.size': 50})

def Plotting(solution, symmet = symmet, nConnect = nConnect,  linestyle = "-", legend = False, trajectory = True, colourcode = True, colourbar = False, mainmarkercolor = "black", markersize = 25, forces = 0, showforce = False): # TODO 
    '''
    solution: Output of solve_ivp
    symmet: number of nodes
    n: number of neighbours connected on each side per node
    linestyle (default: "-"): Linestyle in 1st plot 
    legend (default: False): Whether to show a legend in the 1st plot 
    colourcode (default: True): colourcodes trajectories in 2nd plot if True
    colourbar (default: True): Plots colourbar in 2nd plot if True and if colourcode is True
    mainmarkercolor: Colour of nodes in 2nd plot 
    '''
    viewFrame = -1 # 0 is the first frame, -1 is the last frame
    nFrames = len(solution.t)
    pos = solution.y.T[: , :2*symmet] # positions over time
    vel = solution.y.T[: , 2*symmet:] # velocities over time
    pos2D = np.reshape(pos, (nFrames, symmet, 2))
    vel2D = np.reshape(vel, (nFrames, symmet, 2))
    
    # Position over time
    palette = sns.color_palette("hsv", 2*symmet)
    for i in range(symmet):
        axs[0].plot(solution.t, pos2D[:, i, 0], label = "x" + str(i), linestyle = linestyle, color = palette[i*2])
        axs[0].plot(solution.t, pos2D[:, i, 1], label = "y" + str(i), linestyle = linestyle, color = palette[i*2 + 1])
    if(legend):
        axs[0].legend(loc = 'best')
    axs[0].set(xlabel = 't (a.u.)')
    
    # Nodes at last timestep
    axs[1].plot(pos2D[viewFrame, :symmet, 0], pos2D[viewFrame,:symmet, 1], 
    linestyle = "", marker = "o", color="gray", markerfacecolor = mainmarkercolor, markersize = markersize, zorder = 50)
    
    # Anchor springs
    axs[1].plot([0,0], [0,0], marker = "o", color = "lightgray", markersize = 15)
    for i in range(symmet):
        axs[1].plot((pos2D[viewFrame, i, 0], 0), (pos2D[viewFrame, i, 1], 0),
        linestyle = ":", marker = "", color="lightgray")   
        
    # Radial springs 
    for ni in range(1, nConnect+1): # neighbours to connect to
        for i in range(symmet): # node to connect from 
            axs[1].plot(pos2D[viewFrame, (i, (i+ni)%symmet), 0], pos2D[viewFrame, (i, (i+ni)%symmet), 1], 
            linestyle = ":", marker = "", color="gray")#, linewidth = 5)

    # Trajectories 
    if (trajectory):
        if (colourcode): # Colourcoded trajectory
            ### colourcoding velocities
            normvel = np.zeros((nFrames, symmet)) #nFrames, node
            
            for i in range(symmet):
                for frame in range(nFrames):
                    normvel[frame, i] = np.linalg.norm([vel2D[frame, i, 0], vel2D[frame, i, 1]])
                    
            norm = plt.Normalize(normvel.min(), normvel.max()) 
            
            #####trajectory colorcoded for velocity        
            for i in range(symmet):
                points = pos2D[:, i, :].reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1],points[1:]], axis = 1)
                lc = LineCollection(segments, cmap = 'plasma', norm=norm, zorder = 100)
                lc.set_array(normvel[:, i])
                line = axs[1].add_collection(lc)
        
            if(colourbar):
                axcb = fig.colorbar(line, ax=axs[1])   
                axcb.set_label('velocity (a.u.)')
            
        else: # monochrome trajectory
            for i in range(symmet):
                axs[1].plot(pos2D[:, i, 0], pos2D[:, i, 1], color = "blue", linestyle = "-")

    ### Force vectors
    if(showforce and type(forces) != int):
        forces2d = forces.reshape(symmet, 2)
        for i in range(symmet):
            axs[1].arrow(x = pos2D[0, i, 0], y = pos2D[0, i, 1], 
                         dx = (forces2d[i, 0] - pos2D[0, i, 0]), 
                         dy = (forces2d[i, 1] - pos2D[0, i, 1]),
                         width = 0.7, color="blue")   
            
    axs[1].axis("scaled")
    axs[1].set(xlabel = "x (nm)", ylabel = "y (nm)")  
    plt.tight_layout()

def Plotforces(fcoords, initcoords):  
    fcoords = np.concatenate(fcoords)
    initcoords = np.concatenate(initcoords)
    allnodes = len(initcoords)
    
    fig, ax1 = plt.subplots(1,1, figsize = (10, 10))
    
    for i in range(allnodes):#int(nodes/2)):
        ax1.arrow(x = initcoords[i,0], y = initcoords[i,1], 
        dx = (fcoords[i,0] - initcoords[i,0]), dy = (fcoords[i,1] - initcoords[i,1]),
        width = 0.7, color="blue") 
    plt.axis("scaled")


Plotforces(fcoords, initcoords)


showforces = False
trajectories = False 
fig, axs = plt.subplots(2, 1, figsize = (15, 26))

#Plotting(solution, colourbar = False, mainmarkercolor="darkgray", legend = False)#, forces = fcoords3, showforce = showforces, trajectory=trajectories)#, markersize = 30)
             

Plotting(solution[0], mainmarkercolor="darkgray", legend = True, colourcode=False)#, forces = fcoords3, showforce = showforces, trajectory=trajectories)#, markersize = 30)
Plotting(solution[1], linestyle="--", mainmarkercolor="darkgray")#, forces = fcoords4, showforce = showforces, trajectory=trajectories)
Plotting(solution[2], showforce= showforces, trajectory = trajectories, colourbar = True)
Plotting(solution[3], linestyle="--")#, forces = fcoords2, showforce = showforces, trajectory = trajectories)




## 3D plot
viewFrame = 0 # 0 is the first frame, -1 is the last frame
fig = plt.figure(figsize = (15,15))
ax = fig.add_subplot(111, projection='3d')
zdist = 50

ax.scatter(Pos2D(solution[0])[viewFrame, : ,0], Pos2D(solution[0])[viewFrame, :,1], s = 200, c = "black")
ax.scatter(Pos2D(solution[1])[viewFrame, : ,0], Pos2D(solution[1])[viewFrame, :,1], s = 200, c = "black")
ax.scatter(Pos2D(solution[2])[viewFrame, : ,0], Pos2D(solution[2])[viewFrame, :,1], zdist, s = 200, c = "gray")
ax.scatter(Pos2D(solution[3])[viewFrame, : ,0], Pos2D(solution[3])[viewFrame, :,1], zdist, s = 200, c = "gray")

#3D arrows
# forces2d = fcoords[0].reshape(symmet, 2)
# forces2d2 = fcoords[1].reshape(symmet, 2)
# forces2d3 = fcoords[2].reshape(symmet, 2)
# forces2d4 = fcoords[3].reshape(symmet, 2)
zdists = [0, 0, 50, 50]
linewidth = 3
normalize = True

for ring in range(nRings):
    for i in range(symmet):
        ax.quiver(Pos2D(solution[ring])[0,i,0], Pos2D(solution[ring])[0,i,1], zdists[ring], fcoords[ring][i][0], fcoords[ring][i][1], 0, length = randfs[0][i], normalize = normalize, linewidth = linewidth , edgecolor = "blue")#(forces2d[i,0] - solplot2D0[0,i,0]),(forces2d[i,1] - solplot2D0[0,i,1]), 0)   

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
    



with open('/home/maria/Documents/NPCPython/NPCexampleposter2.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for i in range(symmet):
        spamwriter.writerow(np.append(Pos2D(solution[0])[-1,i], 0))
    for i in range(symmet):
        spamwriter.writerow(np.append(Pos2D(solution[1])[-1,i], 0))
    for i in range(symmet):
        spamwriter.writerow(np.append(Pos2D(solution[2])[-1,i], zdist))
    for i in range(symmet):
        spamwriter.writerow(np.append(Pos2D(solution[3])[-1,i], zdist))




#plt.scatter(F[:,0], F[:,1]) 
#initcoord = np.array([1.,0,np.sqrt(2)/2, np.sqrt(2)/2,0,1,-np.sqrt(2)/2,np.sqrt(2)/2,-1,0,-np.sqrt(2)/2,-np.sqrt(2)/2,0,-1,np.sqrt(2)/2,-np.sqrt(2)/2]) #TODO remove
###### TODO
global xy
global xy1
global xy2
global xy3
global xyA
# xy = solplot2D0[:, np.append(np.arange(symmet), 0), :] # TODO 
# xy1 = solplot2D1[:, np.append(np.arange(symmet), 0), :] # TODO 
# xy2 = solplot2D2[:, np.append(np.arange(symmet), 0), :] # TODO 
# xy3 = solplot2D3[:, np.append(np.arange(symmet), 0), :] # TODO 

xy = Pos2D(solution[0])[:, np.append(np.arange(symmet), 0)] # TODO 
xy1 = Pos2D(solution[1])[:, np.append(np.arange(symmet), 0)] # TODO 
xy2 = Pos2D(solution[2])[:, np.append(np.arange(symmet), 0)] # TODO 
xy3 = Pos2D(solution[3])[:, np.append(np.arange(symmet), 0)] # TODO 

# xy = solplot2D0[:, :symmet, :] # TODO 
# xy1 = solplot2D1[:, :symmet, :] # TODO 
# xy2 = solplot2D2[:, :symmet, :] # TODO 
# xy3 = solplot2D3[:, :symmet, :] # TODO 

xyA = np.zeros((len(xy), 2*symmet, 2))
for i in range(len(xy)):
    xyA[i,:,0] = np.insert(xy[i,:symmet,0], np.arange(symmet), 0)
    xyA[i,:,1] = np.insert(xy[i,:symmet,1], np.arange(symmet), 0)

#%matplotlib qt # Run in console if python doesn't show plot 
class AnimatedScatter(object):
    """An animated scatter plot using matplotlib.animations.FuncAnimation."""
    def __init__(self, xy, numpoints=symmet+1):
        self.numpoints = numpoints
        self.stream = self.data_stream(xy)
        # Setup the figure and axes...
        self.fig, self.ax = plt.subplots(figsize = (9, 10))
        plt.rcParams.update({'font.size': 20})
        # Then setup FuncAnimation.
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=15, frames = len(xy), #TODO change frames back to nFrames?
                                          init_func=self.setup_plot, blit=True)
        #HTML(self.ani.to_html5_video())
        self.ani.save("Damping0.mp4", dpi = 250)

    def setup_plot(self):
        """Initial drawing of the scatter plot."""
#        x, y, x1, y1, x2, y2, x3, y3 = next(self.stream).T
        # 4 to plot x-y coords
        # 4 to plot anchor springs 
        # n * symmet * rings to plot circumferential spring

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
            for i in range(len(xy)):
                x_y = xy[i]
                x_y1 = xy1[i]
                x_y2 = xy2[i]
                x_y3 = xy3[i]
                yield np.c_[x_y[:,0], x_y[:,1], x_y1[:,0], x_y1[:,1], x_y2[:,0], x_y2[:,1], x_y3[:,0], x_y3[:,1]]

    def update(self, i):
        """Update the plot."""
        x, y, x1, y1, x2, y2, x3, y3 = next(self.stream).T
        

        xa = np.insert(x[:symmet], np.arange(symmet), 0)
        ya = np.insert(y[:symmet], np.arange(symmet), 0)
        x1a = np.insert(x1[:symmet], np.arange(symmet), 0)
        y1a = np.insert(y1[:symmet], np.arange(symmet), 0)       
        x2a = np.insert(x2[:symmet], np.arange(symmet), 0)
        y2a = np.insert(y2[:symmet], np.arange(symmet), 0)        
        x3a = np.insert(x3[:symmet], np.arange(symmet), 0)
        y3a = np.insert(y3[:symmet], np.arange(symmet), 0)
        


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
                for i in range(symmet): # node to connect from 
                    self.lines[count][0].set_data((xlist[lnum][i], xlist[lnum][(i+ni)%symmet]), (ylist[lnum][i], ylist[lnum][(i+ni)%symmet]))
                    count += 1
        
        return [self.lines[i][0] for i in range(int(8 + 8*2*4))]#self.lines[0][0], self.lines[1][0], self.lines[2][0], self.lines[3][0], self.lines[4][0], self.lines[5][0],self.lines[6][0], self.lines[7][0], #self.line, self.line1, self.line2, self.line3, self.lineA, self.lineA1, self.lineA2, self.lineA3,


if __name__ == '__main__':
    a = AnimatedScatter(xy)

    plt.show()

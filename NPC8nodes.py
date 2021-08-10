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
#import timeit
import seaborn as sns
import csv
import matplotlib.animation as animation
from sklearn.gaussian_process.kernels import RBF
#from IPython.display import HTML
#from matplotlib import rc
#from matplotlib.patches import FancyArrowPatch

### Parameters
symmet = 8      # Rotational symmetry of the NPC
mag = 25        # Magnitude of deformation [nm]; 3 standard deviation -> 99.7 % of forces on a node lie within this range
nConnect = 2    # Number of connected neighbour nodes in clock-wise and anti-clockwise direction

r = [50, 54, 54, 50]
ringAngles = [0, 0.2069, 0.2407, 0.4476]
z = [0, 0, 50, 50]
nRings = len(r)     # Number of rings. 

class DeformNPC:
    def __init__(self, nConnect = 2, mag = 25, symmet = 8, r = 0, ringAngles = 0, z = 0):
        '''
        Models deformed NPCs using solve_ivp based on a simple, rotationally symmetric node and spring model with deforming forces applied in xy direcion. 
        ### Input ###
        nConnect: Number of connected neighbour nodes in clock-wise and anti-clockwise direction
        mag: Magnitude of deformation [nm]. Number represents 3 standard deviations -> 99.7 % of forces on a node lie within this range
        symmet: Symmetry of the NPC (Default 8 )
        nRings: Number of NPC rings 
        r: Radius of NPC rings assuming 8-fold symmetry. Must be a list of length nRings
        ringAngles: Rotational angle offset of NPC rings. Must be a list of length nRings
        z: z position of NPC rings. Stays unchanged. Must be a list of length nRings
        
        ### Relevant output ###
        solution: solve_ivp output for all NPC rings 
        fcoords: Coordinates of force vectors for all NPC rings
        initcoords: starting coordinates 
        randfs: list of magnitude of forces for all NPC rings 
        r: Radius of NPC rings corrected for symmetry (i.e. increasing with symmetry)
        z: z position of NPC rings
       
        '''
        self.solution = [] 
        self.symmet = symmet # Rotational symmetry of the NPC
        self.initcoords = [] # starting coordinates of nodes    
        self.fcoords = []    # Coordinates of forces 
        self.z = z # z coordinates (stays unchanged)
        nRings = len(r)
        damp = 1 # damping
        kr = 0.7 # spring constant of anchor spring 
        
        tlast = 40
        tspan = [0,tlast]      
        #teval = np.arange(0, tlast, 0.2) # needed for animation function
        teval = None        
        
        Lrests = [] # circulant matrix of resting spring lengths
        Ks = [] # circulant matrix of spring constants 
        y0s = [] # initial coordinates and velocities per node

        if(nConnect > self.symmet/2):
            nConnect = int(np.floor(self.symmet/2))
            warn("Selected number of neighbours nConnect too large. nConnect has been changed to " + str(nConnect) + ".")
        
        if(len(ringAngles) != nRings or len(z) != nRings):
            warn("r, ringAngles, and z must be of length nRings: " + str(nRings))

        for i in range(nRings):  
            r[i] = self.adjustRadius(r[i])
            initcoord = self.initialcoords(r[i], ringAngle = ringAngles[i])
            self.initcoords.append(initcoord) 
            Lrests.append(self.springlengths(initcoord)) 
            Ks.append(self.springconstants())
            y0s.append(np.concatenate((initcoord.flatten(), np.zeros(2 * self.symmet)))) 
 
        self.randfs = self.forcesMultivariateNorm(self.initcoords, r, mag, nRings = nRings) # generate random forces
  
        # Solve ODE, ring 1 - 4       
        for i in range(nRings):
            self.solution.append(solve_ivp(self.npc, tspan, y0s[i], t_eval=teval, 
                                           method='RK45', args=(r[i], Lrests[i], 
                                        Ks[i], kr, self.randfs[i], damp, nConnect)))   
            
            self.fcoords.append(self.initialcoords(r[i], ringAngles[i], self.randfs[i]))
        
        self.r = r    
            
    ### Methods ###


    def npc(self, t, y, r, Lrest, K, kr, randf, damp, nConnect):
        '''
        t: time points 
        y: values of the solution at t 
        r: radius of NPC ring
        Lrest: Circulant matrix of resting lengths of all springs 
        K: Circulant matrix of all radial spring constants 
        kr: Spring constants of anchor springs 
        randf: array of forces (length = symmet) to be applied in radial direction to each node 
        damp: Damping factor 
        nConnect: Number of connected neighbours in cw and ccw direction for each node 

        output: solutions at t. x and y components of positions and velocities of each node for each timestep 
        '''
        v = np.reshape(y[2*self.symmet:], (self.symmet, 2))
        x = np.reshape(y[:2*self.symmet], (self.symmet, 2))
        
        anc = np.array([0., 0.]) # coordinates of anchor node   
        F = np.zeros((self.symmet, 2)) # Forces
        
        for i in range(self.symmet): 
            F[i] = randf[i]*x[i] / np.linalg.norm([x[i], anc]) 
    
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
    
    
    def adjustRadius(self, r8):
        """Adjusts radius r with symmetry. No adjustment is made when symmetry is 8. Radius is viewed
        as the length of the symmetric side of an isoceles triangle whose tip (angle alpha) points towards the 
        center of the NPC and whose base is the section between two neighbouring nodes at the circumference. Think slice of cake.
        # Input: 
        r8: radius of a default 8-fold symmetrical NPC
        
        ## Output: 
        radius of NPC with symmetry equal to symmet (rnew = r8 if symmet = 8)
        
        """
        alpha = 2*np.pi/self.symmet # Angle at the tip of triangular slice (pointing to center of NPC)
        theta = 0.5 * (np.pi - alpha) # Either angle at the base of (isosceles) triangular slice
        halfbase = r8 * np.sin(np.pi/8) # half the distance between two corners of an NPC ring (= half the base of triangular slice)
        height = halfbase * np.tan(theta) # height from base to tip of triangular slice
        return np.sqrt(height**2 + halfbase**2) # new radius 
               
      
    def pol2cart(self, rho, phi):
        '''Transforms polar coordinates of a point (rho: radius, phi: angle) to 2D cartesian coordinates.
        '''
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return(x, y)
    
    
    def initialcoords(self, r, ringAngle = 0, forces = 0): 
        '''
        Generates cartesian coordinates of the NPC given radius and self.symmet 
        ## Input ##
        r: NPC Radius
        ringAngle: Angular offset of ring (default 0)
        forces (optional): input 1D forcevector to view coordinates of forces; used for visualisation

        ## Return values ##
        2D Cartesian coordinates 
        '''
        
        if(type(forces) != int):
            if (len(forces) != self.symmet):
                warn("forces must be 0 or an array with len(self.symmet")
                
        forces = forces * np.ones(self.symmet) # forces is 0 or np.array with len(self.symmet)
        rotAngle = 0.

        initcoord = np.zeros((self.symmet, 2))
        
        for i in range(self.symmet):
            initcoord[i, 0], initcoord[i, 1] = self.pol2cart(r + forces[i], 
                                                             rotAngle+ringAngle)
            rotAngle += 2*np.pi/self.symmet

        return initcoord
    
    
    def springlengths(self, initcoord): 
        '''Compute lengths of springs from coordinates and returns circulant matrix
        '''
        l = np.zeros(self.symmet)
        for i in range(self.symmet):
            l[i] = np.linalg.norm(initcoord[0, :] - initcoord[i, :])      
        return circulant(l)
    
    
    def springconstants(self): # TODO: More variable spring constants? 
        "Returns circulant matrix of spring constants "
        k = np.ones(int(np.floor(self.symmet/2)))
        if(self.symmet%2 == 0): #if symmet is even
            k[-1] = k[-1]/2 # springs that connect opposite corners will be double-counted. Spring constant thus halved 
            K = circulant(np.append(0, np.append(k, np.flip(k[:-1]))))
        else: #if symmet is odd 
            K = circulant(np.append(0, [k, np.flip(k)]))
        return K
    
    
    def forcesMultivariateNorm(self, initcoords, r, mag, nRings = 1): # TODO: include distances in z
        '''
        Returns array of Forces that are later applied in radial direction to the NPC corners
        ## Input ## 
        *coordring: Initial coordinates of nodes for an arbitrary number of rings. 
        mag: Total magnitude of distortion. Default is 50. 
        r: radius of all NPC rings 
        nRings: Number of NPC rings (default 1)
        ## Returns ## 
        For each ring, an array of forces applied to each node
        '''

        nodesTotal = self.symmet*nRings # total number of nodes over all rings 
        
        allcoords = np.asarray(initcoords) # separated by NPC rings
        allcoords = allcoords.reshape(nodesTotal, 2) # not separated by NPC rings 
        
        sigma = np.min(r) # free parameter of RBF kernel that outputs covariance of forces on two nodes based on their euclidean distance

        cov = np.zeros((nodesTotal, nodesTotal)) # covariance matrix based on which random forces on nodes are drawn 


        kernel = 1.0 * RBF(sigma)#(1/(2*sigma**2))
        cov = kernel.__call__(X = allcoords)
            
        mag = (mag/3)**2    # 3*SD to Var
        cov = mag * cov 

        rng = np.random.default_rng() 
        u = rng.standard_normal(len(allcoords))
        L = np.linalg.cholesky(cov) # cholesky decomposition since sampling using rng.multivariate_normal(mean, cov) is numerically unstable
        F = L @ u        
        
        return np.split(F, nRings)


### Instantiate DeformNPC
deformNPC = DeformNPC(nConnect, mag, symmet = symmet, r = r, ringAngles = ringAngles, z = z)
solution = deformNPC.solution
fcoords = deformNPC.fcoords # coordinates of force vectors
initcoords = deformNPC.initcoords # starting coordinates 
randfs = deformNPC.randfs # magnitude of forces
z = deformNPC.z
r = deformNPC.r


### Functions to help with plotting and CSV generation
def Sol2D(solution):
    """
    input: DeformNPC.solution for a given NPC ring
    output: 2D arrays of position and of velocity of nodes in a ring over time [timeframe, node, dimension (x or y)]
    """
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


def ColourcodeZ(z, darkest = 0.1, brightest = 0.5):
    if(type(z) == list):
        return [str(i) for i in np.interp(z, (min(z), max(z)), (darkest, brightest))]
    else: return "0.5"
    

#### Plotting ####################################################################
plt.rcParams.update({'font.size': 30})

def Plot2D(solution, z = z, symmet = symmet, nConnect = nConnect,  linestyle = "-", trajectory = True, colourcode = True, anchorsprings = True, radialsprings = True, markersize = 20, forces = 0, showforce = False): # TODO 
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
        
    #trajectories = False 
    fig, ax = plt.subplots(1, 1, figsize = (10, 10))
    viewFrame = -1 # 0 is the first frame, -1 is the last frame  
    mainmarkercolor = ColourcodeZ(z)
    
    for i in range(nRings):
        
        nFrames = len(solution[i].t)# Nodes at last timestep
        pos2D, vel2D = Sol2D(solution[i])
        
        ax.plot(pos2D[viewFrame, :symmet, 0], pos2D[viewFrame,:symmet, 1], 
        linestyle = "", marker = "o", color="gray", markerfacecolor = mainmarkercolor[i], markersize = markersize, zorder = 50)
        
        if (anchorsprings):
            # Anchor springs
            ax.plot([0,0], [0,0], marker = "o", color = "lightgray", markersize = 15)
            for i in range(symmet):
                ax.plot((pos2D[viewFrame, i, 0], 0), (pos2D[viewFrame, i, 1], 0),
                linestyle = ":", marker = "", color="lightgray")   
            
        # Radial springs 
        if(radialsprings):
            for ni in range(1, nConnect+1): # neighbours to connect to
                for i in range(symmet): # node to connect from 
                    ax.plot(pos2D[viewFrame, (i, (i+ni)%symmet), 0], pos2D[viewFrame, (i, (i+ni)%symmet), 1], 
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
                    line = ax.add_collection(lc) # TODO will only be saved for the last ring TODO: is this TODO still up to date?
                       
            else: # monochrome trajectory
                for i in range(symmet):
                    ax.plot(pos2D[:, i, 0], pos2D[:, i, 1], color = "blue", linestyle = "-")
    
        ### Force vectors
        if(showforce and type(forces) != int):
            forces2d = forces.reshape(symmet, 2)
            for i in range(symmet):
                ax.arrow(x = pos2D[0, i, 0], y = pos2D[0, i, 1], 
                             dx = (forces2d[i, 0] - pos2D[0, i, 0]), 
                             dy = (forces2d[i, 1] - pos2D[0, i, 1]),
                             width = 0.7, color="blue")   
    if(trajectory and colourcode):
        axcb = fig.colorbar(line, ax=ax)   
        axcb.set_label('velocity (a.u.)')
            
    ax.axis("scaled")
    ax.set(xlabel = "x (nm)", ylabel = "y (nm)")  
    plt.tight_layout()


def Plotforces(fcoords, initcoords):  
    fcoords = np.concatenate(fcoords)
    initcoords = np.concatenate(initcoords)
    allnodes = len(initcoords)
    
    fig, ax1 = plt.subplots(1,1, figsize = (10, 10))
    
    for i in range(allnodes):
        ax1.arrow(x = initcoords[i,0], y = initcoords[i,1], 
        dx = (fcoords[i,0] - initcoords[i,0]), dy = (fcoords[i,1] - initcoords[i,1]),
        width = 0.7, color="blue") 
    plt.axis("scaled")



def XYoverTime(solution, symmet = symmet, nRings = nRings, linestyle = "-", legend = False): 
    '''x and y positions over time'''
    fig, ax = plt.subplots(1,1, figsize = (10, 10))
    palette = sns.color_palette("hsv", 2*symmet)
    for ring in range(nRings):
        for i in range(symmet):
            ax.plot(solution[ring].t, Pos2D(solution[ring])[:, i, 0], label = "x" + str(i), linestyle = linestyle, color = palette[i*2])
            ax.plot(solution[ring].t, Pos2D(solution[ring])[:, i, 1], label = "y" + str(i), linestyle = linestyle, color = palette[i*2 + 1])
    if(legend):
        ax.legend(loc = 'best')
    ax.set(xlabel = 't (a.u.)')
    plt.show()
    



def Plot3D(solution, z, symmet, nRings, viewFrame = -1, colour = ["black", "gray"]):
    '''viewFrame: 0 is first frame, -1 is last frame'''
    fig = plt.figure(figsize = (15,15))
    ax = fig.add_subplot(111, projection='3d')

    linewidth = 3
    normalize = True
    
    colour = ColourcodeZ(z)
    
    for ring in range(nRings): 
        ax.scatter(Pos2D(solution[ring])[viewFrame, : ,0], Pos2D(solution[ring])[viewFrame, :,1], z[ring], s = 200, c = colour[ring])
        for node in range(symmet):
            ax.quiver(Pos2D(solution[ring])[0, node, 0], Pos2D(solution[ring])[0, node ,1], z[ring], fcoords[ring][node][0], fcoords[ring][node][1], 0, length = randfs[0][node], normalize = normalize, linewidth = linewidth , edgecolor = "blue") 
    
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

    ax.grid(False)
    plt.show()


def Export2CSV():
    with open('/home/maria/Documents/NPCPython/NPCexampleposter3.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for ring in range(nRings):
            for node in range(symmet):
                spamwriter.writerow(np.append(Pos2D(solution[ring])[-1,node], z[ring]))
    

class AnimatedScatter(object):
    """An animated scatter plot using matplotlib.animations.FuncAnimation."""
    def __init__(self, solution, nRings, nConnect, symmet, r, randfs):
        self.solution = solution
        self.nRings = nRings

        framenumbers = []
        
        for i in range(self.nRings): # check framenumbers are consistent for each ring
            framenumbers.append(len(self.solution[i].t)) 
        if (len(set(framenumbers)) != 1):
            warn("Number of timesteps for all ring must be the same in order to animate deformation.")
            return
        if (nRings != 4):
            warn("Animation function works correctly only for 4 NPC rings at the moment.")
            return
        
        nframes = len(self.solution[0].t)
        
        self.nConnect = nConnect
        self.symmet = symmet
        self.xy = self.xydata()        
        self.stream = self.data_stream(self.xy)
        
        # Setup the figure and axes...
        self.axscale = 1.2 * (np.amax(randfs) + max(r))
        self.fig, self.ax = plt.subplots(figsize = (9, 10))
        plt.rcParams.update({'font.size': 20})
        
        # Then setup FuncAnimation.
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=(5000/nframes), 
                                          init_func=self.setup_plot, blit=True)
        #HTML(self.ani.to_html5_video())
        self.ani.save("Damping0.mp4", dpi = 250)

    def xydata(self):
        xy = []
        for ring in range(self.nRings):
            xy.append(Pos2D(solution[ring])[:, np.append(np.arange(self.symmet), 0)])
        return xy
            
    def setup_plot(self):
        """Initial drawing of the scatter plot."""

        self.lines = []
        for i in range(int(self.nRings*2 + self.symmet*self.nRings*self.nConnect)):   #TODO code for 4 rings only!
            if (i <= 1): # 0, 1: lower rings
                self.lobj = self.ax.plot([], [], marker = "o", color = "gray", markerfacecolor = "gray", linestyle = "", markersize = 15) 
            elif (i > 1 and i <=3): #2, 3 upper rings
                self.lobj = self.ax.plot([], [], marker = "o", color = "gray", markerfacecolor = "black", linestyle = "", markersize = 15) 
            elif (i > 3 and i <= 7): #4, 5, 6, 7: 4 rings to anchor
                self.lobj = self.ax.plot([], [], marker = "", color = "orange", linestyle = "-", zorder = 0) # anchor
            else: # 8 - ? #all circumferential springs
                self.lobj = self.ax.plot([], [], marker = "", color = "blue", linestyle = "-")

            self.lines.append(self.lobj)
        
        self.ax.axis("scaled")
        self.ax.set(xlabel = "x (nm)", ylabel = "y (nm)")  
        self.ax.axis([-self.axscale, self.axscale, -self.axscale, self.axscale])
        
        return [self.lines[i][0] for i in range(int(self.nRings*2 + self.symmet*self.nRings*self.nConnect))]
        
    def data_stream(self, pos):
        x = np.zeros((self.symmet+1, self.nRings))
        y = np.zeros((self.symmet+1, self.nRings))
        while True: 
            for i in range(len(self.xy[0])):
                for ring in range(self.nRings):
                    x[:, ring] = self.xy[ring][i][:, 0]
                    y[:, ring] = self.xy[ring][i][:, 1]
                yield x, y
        
    def update(self, i):
        """Update the plot."""

        x, y = next(self.stream)
        
        xa = np.zeros((2*self.symmet, self.nRings))
        ya = np.zeros((2*self.symmet, self.nRings))     
        
        for ring in range(self.nRings):
            for i in range(1, 2*self.symmet, 2): 
                xa[i, ring] = x[int((i-1)/2), ring]
                ya[i, ring] = y[int((i-1)/2), ring]
        
        xlist = list(x.T) + list(xa.T)
        ylist = list(y.T) + list(ya.T)
                
        for lnum, self.line in enumerate(self.lines):
            if lnum >= len(xlist):
                break
            self.line[0].set_data(xlist[lnum], ylist[lnum]) 

        # TODO code  works only for 4 rings!
        count = len(xlist)
        for lnum in range(self.nRings):
            for ni in range(1, self.nConnect+1): # neighbours to connect to
                for i in range(self.symmet): # node to connect from 
                    self.lines[count][0].set_data((xlist[lnum][i], xlist[lnum][(i+ni)%self.symmet]), (ylist[lnum][i], ylist[lnum][(i+ni)%self.symmet])) 
                    count += 1
        return [self.lines[i][0] for i in range(int(self.nRings*2 + self.symmet*self.nRings*self.nConnect))]

if __name__ == '__main__':
    a = AnimatedScatter(solution, nRings, nConnect, symmet, r, randfs)

    plt.show()


#%matplotlib qt # Run in console if python doesn't show plot 

XYoverTime(solution)
Plotforces(fcoords, initcoords)
Plot2D(solution, anchorsprings=False, radialsprings=False, trajectory=False)    
Plot3D(solution, z, symmet, nRings, viewFrame = -1)#, colour = ["black", "black", "gray", "gray"])
#Export2CSV


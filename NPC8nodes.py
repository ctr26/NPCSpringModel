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
mag = 70        # Magnitude of deformation 
nConnect = 2    # Number of connected neighbour nodes in clock-wise and anti-clockwise direction
nRings = 4      # Number of rings 

class DeformNPC:
    def __init__(self, symmet, nConnect, mag, nRings = 1, r = 0, ringAngles = 0, z = 0):
        self.symmet = symmet
        self.initcoords = []        
        self.solution = []
        self.fcoords = []    
        self.z = z
        
        damp = 1 # damping
        kr = 0.7 # spring constant of anchor spring 
        
        tlast = 40
        tspan = [0,tlast]      
        #teval = np.arange(0, tlast, 0.1)
        teval = None        
        
        global Lrests
        Lrests = []
        Ks = []
        y0s = [] 

        if(nConnect > self.symmet/2):
            nConnect = int(np.floor(self.symmet/2))
            warn("Selected number of neighbours nConnect too large. nConnect has been set to " + str(nConnect) + ".")
        
        if(len(r) != nRings or len(ringAngles) != nRings):
            warn("r and ringAngles must be the same length as nRings")

        for i in range(nRings):  
            r[i] = self.AdjustRadius(r[i])
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
    
    def NPC(self, t, y, Lrest, r, K, kr, randf, damp, nConnect):
        '''
        t: time points 
        y: values of the solution at t 
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
        
        for i in range(self.symmet): # TODO test
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
    
    
    def AdjustRadius(self, r8):
        """Adjusts radius r with symmetry. No adjustment is made when symmetry is 8. Radius is viewed
        as the lenght of the symmetric side of an isoceles triangle whose tip (angle alpha) points towards the 
        center of the NPC and whose base is the section between two nodes of a ring at the circumference of the 
        NPC. Think slice of cake.
        # Input: 
        r8: radius of a default 8-fold symmetrical NPC
        
        ## Output: 
        rnew: radius of NPC with symmetry equal to symmet (rnew = r8 if symmet = 8)
        
        """
        alpha = 2*np.pi/self.symmet # Angle at the tip of triangular slice (pointing to center of NPC)
        theta = 0.5 * (np.pi - alpha) # Angle at the base of triangular slice
        halfbase = r8 * np.sin(np.pi/8) # half the distance between two corners of an NPC ring 
        height = halfbase * np.tan(theta) # height from base to tip of triangular slice
        return np.sqrt(height**2 + halfbase**2) # new radius 
        
               
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
        global allcoords
        global AllD
        global LInv
        global cov
        nodesTotal = self.symmet*nRings # total number of nodes over all rings 
        
        allcoords = np.asarray(initcoords) # separated by NPC rings
        allcoords = allcoords.reshape(nodesTotal, 2) # not separated by NPC rings 
        
        sigma = 50
        AllD = np.zeros((nodesTotal, nodesTotal)) # all distances

        cov2 = np.zeros((nodesTotal, nodesTotal))
        for i in range(nodesTotal):
            for j in range(nodesTotal):
                AllD[i, j] = np.linalg.norm(allcoords[i, :] - allcoords[j, :])
                cov2[i, j] = np.exp(-(AllD[i, j]**2 / (2*sigma**2)))
    

    
        LInv = AllD.max() - AllD # invert distances  

        cov = [] # covariance matrix 
        for i in range(nodesTotal):
            cov.append(list(LInv[i]/AllD.max()))

        mean = list(np.zeros(nodesTotal)) # Mean of the normal distribution
                
        rng = np.random.default_rng() # TODO remove seed
        F = rng.multivariate_normal(mean, cov2)#, size = 1000) # TODO cov2
        
    
        initmag = sum(abs(F))
        
        if (initmag != 0):
            F = nRings*mag/initmag * F
        
        if (nRings) == 1: # TODO: more general return statement 
            return np.split(F, nRings)[0]
        else:
            return np.split(F, nRings)#np.array_split(F, nrings)


r = [50, 54, 54, 50]
ringAngles = [0, 0.2069, 0.2407, 0.4476]
z = [0, 0, 50, 50]

deformNPC = DeformNPC(symmet, nConnect, mag, nRings = nRings, r = r, ringAngles = ringAngles, z = z)
solution = deformNPC.solution
fcoords = deformNPC.fcoords # coordinates of force vectors
initcoords = deformNPC.initcoords # starting coordinates 
randfs = deformNPC.randfs # magnitude of forces
z = deformNPC.z

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


def ColourcodeZ(z, darkest = 0.1, brightest = 0.5):
    if(type(z) == list):
        return [str(i) for i in np.interp(z, (min(z), max(z)), (darkest, brightest))]
    else: return "0.5"
    

#### Plotting ####################################################################
plt.rcParams.update({'font.size': 30})

def Plot2D(solution, z = z, symmet = symmet, nConnect = nConnect,  linestyle = "-", trajectory = True, colourcode = True, markersize = 20, forces = 0, showforce = False): # TODO 
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
    #mainmarkercolor = ["darkgray", "darkgray", "black", "black"]
    #mainmarkercolor = ["0.2", "0.2", "0", "0"]
    
    mainmarkercolor = ColourcodeZ(z)
    
    for ring in range(nRings):
        
        nFrames = len(solution[ring].t)# Nodes at last timestep
        pos2D, vel2D = Sol2D(solution[ring])
        
        ax.plot(pos2D[viewFrame, :symmet, 0], pos2D[viewFrame,:symmet, 1], 
        linestyle = "", marker = "o", color="gray", markerfacecolor = mainmarkercolor[ring], markersize = markersize, zorder = 50)
        
        # Anchor springs
        ax.plot([0,0], [0,0], marker = "o", color = "lightgray", markersize = 15)
        for i in range(symmet):
            ax.plot((pos2D[viewFrame, i, 0], 0), (pos2D[viewFrame, i, 1], 0),
            linestyle = ":", marker = "", color="lightgray")   
            
        # Radial springs 
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
                    line = ax.add_collection(lc) # TODO will only be saved for the last ring 
                       
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
    if(colourcode):
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
    
    for i in range(allnodes):#int(nodes/2)):
        ax1.arrow(x = initcoords[i,0], y = initcoords[i,1], 
        dx = (fcoords[i,0] - initcoords[i,0]), dy = (fcoords[i,1] - initcoords[i,1]),
        width = 0.7, color="blue") 
    plt.axis("scaled")



def XYoverTime(solution, symmet = symmet, nRings = nRings, linestyle = "-", legend = False):
    
    fig, ax = plt.subplots(1,1, figsize = (10, 10))
    # Position over time
    palette = sns.color_palette("hsv", 2*symmet)
    for ring in range(nRings):
        for i in range(symmet):
            ax.plot(solution[ring].t, Pos2D(solution[ring])[:, i, 0], label = "x" + str(i), linestyle = linestyle, color = palette[i*2])
            ax.plot(solution[ring].t, Pos2D(solution[ring])[:, i, 1], label = "y" + str(i), linestyle = linestyle, color = palette[i*2 + 1])
    if(legend):
        ax.legend(loc = 'best')
    ax.set(xlabel = 't (a.u.)')
    plt.show()
    



def Plot3D(solution, z, symmet, nRings, viewFrame = 0, colour = ["black", "gray"]):
## 3D plot
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
    
    # Bonus: To get rid of the grid as well:
    ax.grid(False)
    plt.show()





XYoverTime(solution)
Plotforces(fcoords, initcoords)
Plot2D(solution)    
Plot3D(solution, z, symmet, nRings, viewFrame = -1)#, colour = ["black", "black", "gray", "gray"])


def Export2CSV():
    with open('/home/maria/Documents/NPCPython/NPCexampleposter3.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for ring in range(nRings):
            for node in range(symmet):
                spamwriter.writerow(np.append(Pos2D(solution[ring])[-1,node], z[ring]))
    
Export2CSV


#%matplotlib qt # Run in console if python doesn't show plot 
class AnimatedScatter(object):
    """An animated scatter plot using matplotlib.animations.FuncAnimation."""
    def __init__(self, solution, nRings, numpoints = symmet+1):
        self.solution = solution
        
        framenumbers = []
        for i in range(nRings):
            framenumbers.append(len(self.solution[i].t))
        if (len(set(framenumbers)) != 1):
            warn("Number of timesteps for all ring must be the same in order to animate deformation.")
            return
    
        self.nRings = nRings
        self.numpoints = numpoints
        self.xy = self.xydata()
        
        self.stream = self.data_stream(self.xy)
        # Setup the figure and axes...
        self.fig, self.ax = plt.subplots(figsize = (9, 10))
        plt.rcParams.update({'font.size': 20})
        # Then setup FuncAnimation.
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=15, frames = len(self.xy), #TODO change frames back to nFrames?
                                          init_func=self.setup_plot, blit=True)
        #HTML(self.ani.to_html5_video())
        self.ani.save("Damping0.mp4", dpi = 250)

    def xydata(self):
        xy = []
        for ring in range(self.nRings):
            xy.append(Pos2D(solution[ring])[:, np.append(np.arange(symmet), 0)])
        return xy
            
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
        x = np.zeros((symmet+1, self.nRings))
        y = np.zeros((symmet+1, self.nRings))
        while True: 
            for i in range(len(self.xy[0])):
                for ring in range(self.nRings):
                    x[:, ring] = self.xy[ring][i][:, 0]
                    y[:, ring] = self.xy[ring][i][:, 1]
                yield x, y
        
    def update(self, i):
        """Update the plot."""

        x, y = next(self.stream)
        
        xa = np.zeros((2*symmet, self.nRings))
        ya = np.zeros((2*symmet, self.nRings))     
        
        for ring in range(nRings):
            for i in range(1, 2*symmet, 2): 
                xa[i, ring] = x[int((i-1)/2), ring]
                ya[i, ring] = y[int((i-1)/2), ring]
        
        xlist = list(x.T) + list(xa.T)
        ylist = list(y.T) + list(ya.T)
                
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
    a = AnimatedScatter(solution, nRings)

    plt.show()
#AnimatedScatter(xy)
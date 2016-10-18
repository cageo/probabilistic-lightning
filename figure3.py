#------------------------------------------------------------------------------
# ABOUT figure3.py
#------------------------------------------------------------------------------

# This script is intended to demonstrate the process of developing a 
# probabilistic lightning strike density surface by creating Figure 3 in the 
# paper.

# Built with: Python 2.7.11
#             NumPy 1.10.4
#             Matplotlib 1.5.1

#------------------------------------------------------------------------------
# LICENSING
#------------------------------------------------------------------------------

# The MIT License (MIT)

# Copyright (c) 2015 Thomas R. Etherington and George L.W. Perry

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

#------------------------------------------------------------------------------
# 1. IMPORT PACKAGES AND SET RANDOM SEED
#------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import patches

#------------------------------------------------------------------------------
# 2. CREATE EXAMPLE DATA
#------------------------------------------------------------------------------

# Specify spatial extent and resolution
nRow = 100
nCol = 100
cell = 100.0
minX = 1755000.0
minY = 5915000.0
maxX = minX + (nCol * cell)
maxY = minY + (nRow * cell)

# Create an area of interest 'land' array
landArray = np.zeros((nRow,nCol))
landArray[20:80, 20:80] = 1
np.place(landArray, landArray==0, np.nan)

# Create lightning strikes - x, y, angle, semi-major, semi-minor
strikeData = [[1760400, 5921200, 80, 2.0, 0.8],
              [1757500, 5918000, 45, 1.5, 1.0],
              [1764000, 5922000, 135, 3.0, 1.0],
              [1760500, 5916000, 85, 1.2, 0.9],
              [1757500, 5922500, 45, 3.0, 1.0],
              [1760000, 5920000, 10, 2.0, 1.0]]

#------------------------------------------------------------------------------
# 3. CALCULATE STRIKE PROBABILITIES
#------------------------------------------------------------------------------

strikeGrid = np.zeros((nRow, nCol))

i = -1
for strike in strikeData:
    i = i + 1
    
    x = strike[0]
    y = strike[1]
    # Convert xy coordinate to row and column index
    row = nRow - int((float(y) - minY) / cell) - 1
    col = int((float(x) - minX) / cell)
    # Extract 50% probability ellipse information
    angle = strike[2] - 90 # -90 to convert to required origin    
    semimajor = max(strike[3] * 1000, cell) # converting from km to m
    semiminor = max(strike[4] * 1000, cell) # converting from km to m

    #--------------------------------------------------------------------------

    # Convert values as required
    semimajorCell = int(round(semimajor / cell, 0))
    semiminorCell = int(round(semiminor / cell, 0))
    semiAxis50toSigma = 1.177
    sigmaMajor = int(round(semimajor / semiAxis50toSigma / cell, 0))
    sigmaMinor = int(round(semiminor / semiAxis50toSigma / cell, 0))
    rad = np.deg2rad(angle)

    # Determine the dimensions required
    maxDim = max(semimajorCell, semiminorCell) * 3 # 3 to get 99.9% of gaussian distribution
    dim = 1 + (maxDim * 2)
    
    #--------------------------------------------------------------------------
    
    # Create a two dimensional Gaussian surface for the strike data
    
    # Create grids for X and Y of desired size with central row and column = 0
    x = np.arange(-maxDim, maxDim + 1)
    y = np.arange(-maxDim, maxDim + 1)
    X, Y = np.meshgrid(x, y)
    
    # Apply equation 1 in Bourscheidt et al. (2014)
    a = ((np.cos(rad) ** 2) / (2 * sigmaMajor ** 2)) + (np.sin(rad) ** 2 / (2 * sigmaMinor ** 2))
    b = ((np.sin(2 * rad)) / (4 * sigmaMajor ** 2)) - ((np.sin(2 * rad)) / (4 * sigmaMinor ** 2))
    c = ((np.sin(rad) ** 2) / (2 * sigmaMajor ** 2)) + (np.cos(rad) ** 2 / (2 * sigmaMinor ** 2))
    e1 = 1.0 / (2 * np.pi * sigmaMajor * sigmaMinor)
    e2 = (a * (X - 0) ** 2) + (2 * b * (X - 0) * (Y - 0)) + (c * (Y - 0) ** 2)
    g2D = np.round(e1 * np.exp(-e2), 6)

    #--------------------------------------------------------------------------

    # Determine the slice of the ellipse to remove ellipse outside of grid extent
    minRadj = max(maxDim - row, 0)
    minCadj = max(maxDim - col, 0)
    maxRadj = max(row + maxDim + 1 - strikeGrid.shape[0], 0)
    maxCadj = max(col + maxDim + 1 - strikeGrid.shape[1], 0)
    g2DSlice = g2D[0+minRadj:dim-maxRadj, 0+minCadj:dim-maxCadj]
    
    # Determine the slice of the grid
    minRg = max(row - maxDim, 0)
    minCg = max(col - maxDim, 0)
    maxRg = min(row + maxDim + 1, strikeGrid.shape[0])
    maxCg = min(col + maxDim + 1, strikeGrid.shape[1])
    
    # Add ellipse to main grid
    strikeGrid[minRg:maxRg, minCg:maxCg] = strikeGrid[minRg:maxRg, minCg:maxCg] + g2DSlice

    # Determine the total probabiity on land
    pLand = np.nansum(landArray[minRg:maxRg, minCg:maxCg] * g2DSlice)
    strike.append('{0:.6f}'.format(pLand))

#------------------------------------------------------------------------------
# 4. PLOT FIGURE
#------------------------------------------------------------------------------

# Set the default font to compuet modern to match the math font
#mpl.rc('font', family = 'serif', serif = 'cmr10') # LaTeX style
mpl.rc('font', **{'family':'sans-serif','sans-serif':['Arial'],
                  'style':'normal'}) # arial
mpl.rc('font', size=8)
# Set the defualt for axes line widths
mpl.rc('axes', linewidth=0.5)

# Set the figure size
fig = plt.figure(2, figsize=(190/25.4, 100/25.4))

#------------------------------------------------------------------------------

# Make items for the legends
black_cross, = plt.plot((), "kx", markersize=4, scalex=False, scaley=False)

# Make handler thingies in order to be able to create a square and ellipse in the legend
from matplotlib.legend_handler import HandlerPatch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Square
class AnyObject(object):
    pass

class AnyObjectHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent # we don't need x0 as we are creating a square
        width, height = handlebox.width, handlebox.height
        patch = patches.Rectangle([(width/2)-height/2, y0], height, height, linewidth=1, fill=False, linestyle='solid', color='k', transform=handlebox.get_transform())
        handlebox.add_artist(patch)
        return patch

# Ellipse
class HandlerEllipse(HandlerPatch):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
        p = mpatches.Ellipse(xy=center, width=width + xdescent,
                             height=height + ydescent)
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]

ellipseLegend = mpatches.Circle((0.5, 0.5), 0.25, linewidth=1, linestyle='dotted', fill=False, color='k')

#------------------------------------------------------------------------------
# Part (a) Single strike example
#------------------------------------------------------------------------------

plt.subplot(1,3,1)
plt.xticks(np.arange(0))
plt.yticks(np.arange(0))

# Plot probability surface

# Multiply by 100 to convert strikes per hectre to strikes per km2
plt.contourf(g2D * 100, levels=[0, 0.024, 0.048, 0.072, 0.096, 0.120], 
             cmap=plt.cm.RdPu_r, extent=[minX,maxX,minY,maxY], origin='image')
plt.axis('off')

cbar = plt.colorbar(shrink=1, pad=0.05, orientation='horizontal')
cbar.outline.remove() 
cbar.ax.tick_params(direction='out')
cbar.ax.set_xlabel(r"Single strike density ($\mathregular{km^{-2}}$)")

# Plot best location and error ellipse
x = strikeData[5][0]
y = strikeData[5][1]
angle = - strikeData[5][2] + 90 # to convert to patches ellipse requirements
major = strikeData[5][3] * 2000 # to convert semimajor in km to major in m
minor = strikeData[5][4] * 2000 # to convert semiminor in km to minor in m
pLand = strike[5]

ax = fig.gca()
ax.set_aspect('equal') # to keep the array square 

ax.text(0, 1, '(a)',
        horizontalalignment='right',
        verticalalignment='bottom',
        transform=ax.transAxes)
ax.text(0.5, 1, r'$\mathregular{\Sigma \mathit{p} = 0.92}$',
        horizontalalignment='center',
        verticalalignment='bottom',
        transform=ax.transAxes)

r = patches.Rectangle((1757000, 5917000), 6000, 6000, linewidth=1, fill=False, linestyle='solid', color='k')
ax.add_artist(r)

ax.plot(x, y, "kx", markersize=4, scalex=False, scaley=False)
ax.text(x, y - 500, pLand[:4], fontsize=6, 
        verticalalignment='center', horizontalalignment='center')
e = patches.Ellipse((x, y), major, minor, angle=angle, linewidth=1, linestyle='dotted', fill=False, color='k')
ax.add_artist(e)

#------------------------------------------------------------------------------
# Part (b) Multi-strike example
#------------------------------------------------------------------------------

plt.subplot(1,3,2)
plt.xticks(np.arange(0))
plt.yticks(np.arange(0))

# Plot probability surface
# Multiply by 100 to convert strikes per hectre to strikes per km2
plt.contourf(strikeGrid * 100, levels=[0, 0.05, 0.10, 0.15, 0.20, 0.25], 
             cmap=plt.cm.RdPu_r, extent=[minX,maxX,minY,maxY], origin='image')
plt.axis('off')

cbar = plt.colorbar(shrink=1, pad=0.05, orientation='horizontal')
cbar.outline.remove() 
cbar.ax.tick_params(direction='out')
cbar.ax.set_xlabel(r"Total strike density ($\mathregular{km^{-2}}$)")


# Add area of interest rectangle
ax = fig.gca()
ax.set_aspect('equal') # to keep the array square

ax.text(0, 1, '(b)',
        horizontalalignment='right',
        verticalalignment='bottom',
        transform=ax.transAxes)
ax.text(0.5, 1, r'$\mathregular{\Sigma \mathit{p} = 2.85}$',
        horizontalalignment='center',
        verticalalignment='bottom',
        transform=ax.transAxes)
        
r = patches.Rectangle((1757000, 5917000), 6000, 6000, linewidth=1, fill=False, linestyle='solid', color='k')
ax.add_artist(r)

# Plot strike locations and ellipses
for strike in strikeData:
    x = strike[0]
    y = strike[1]
    angle = - strike[2] + 90 # to convert to patches ellipse requirements
    major = strike[3] * 2000 # to convert semimajor in km to major in m
    minor = strike[4] * 2000 # to convert semiminor in km to minor in m
    pLand = strike[5]
    ax.plot(x, y, "kx", markersize=4, scalex=False, scaley=False)
    ax.text(x, y - 500, pLand[:4], fontsize=6, 
            verticalalignment='center', horizontalalignment='center')
    
    e = patches.Ellipse((x, y), major, minor, angle=angle, linewidth=1, linestyle='dotted', fill=False, color='k')
    ax.add_artist(e)

#------------------------------------------------------------------------------
# Legend
#------------------------------------------------------------------------------

plt.subplot(1,3,3)

plt.axis('off')
ax = fig.gca()
ax.set_aspect('equal') # to keep the array square

# Add legend
plt.legend([black_cross, AnyObject(), ellipseLegend],
           ["Lightning strike, with \n probability of being \n within area of interest", "Area of interest", "Strike 50 % probability \n ellipse"],
           handler_map={AnyObject: AnyObjectHandler(), mpatches.Circle: HandlerEllipse()},
           numpoints=1, loc='center', frameon= False, ncol=1, prop={'size':8})

#------------------------------------------------------------------------------

fig.savefig('figure3.png', dpi=150, bbox_inches='tight', pad_inches = 0.1)

#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# ABOUT figure4.py
#------------------------------------------------------------------------------

# This script is intended to demonstrate the process of adaptively smoothing
# a probabilistic lightning strike density surface by creating Figure 4 in the 
# paper.

# Built with: Python 2.7.11
#             NumPy 1.10.4
#             SciPy 0.17.0
#             NLMpy 0.1.1
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
from scipy.ndimage import filters
from scipy import ndimage
import nlmpy
np.random.seed(0) # to create same results

#------------------------------------------------------------------------------
# 2. CREATE LIGHTNING SURFACE
#------------------------------------------------------------------------------

# Specify spatial extent and resolution
nRow = 500
nCol = 500
cell = 100.0
minX = 1360000.0
minY = 5135000.0
maxX = minX + (nCol * cell)
maxY = minY + (nRow * cell)

# Create a gradient and fractal surface
gradient = nlmpy.planarGradient(nRow, nCol, direction=135)
mpd = nlmpy.mpd(nRow, nCol, 0.1)
# Blend arays togather to create a hypothetical lightning surface
lightningArray = nlmpy.blendArray(gradient, [mpd], [0.9, 0.1])

#------------------------------------------------------------------------------
# 3. CREATE AN AREA OF INTEREST AND CALCULATE ITS SUMMARY VALUES
#------------------------------------------------------------------------------

# Create an area of interest 'land' array
landArray = np.zeros((nRow,nCol))
landArray[100:400, 100:400] = 1
np.place(landArray, landArray==0, np.nan)

# Get summary values for the counts that are in mostly land cells
masked = lightningArray * landArray
mean = np.nanmean(masked)
variance = np.nanvar(masked)
n = np.nansum(landArray)

#------------------------------------------------------------------------------
# 4. SMOOTH SURFACE BASED ON SPATIAL AUTOCORRELATION
#------------------------------------------------------------------------------

# Specify the range of distances to analyse
minDistance = 5
maxDistance = 25
# Create empty lists to hold the results for each distance
zScores = []
meanSmoothed = []

# For each distance
for d in range(minDistance, maxDistance + 1):
    
    print(d)

    # Create a circular window of radius = d
    windowDimension = (d * 2) + 1
    binaryWindow = np.ones((windowDimension, windowDimension))
    binaryWindow[d, d] = 0
    distance = ndimage.distance_transform_edt(binaryWindow)
    binaryWindow = distance <= d
    
    # Calcualte the mean value within the window for each cell
    sumWindow = filters.convolve(lightningArray, binaryWindow, mode='constant')
    n_d = np.sum(binaryWindow)
    meanWindow = sumWindow / n_d
    meanSmoothed.append(meanWindow)

    # Calculate the Getis-Ord z score
    numerator = sumWindow - n_d * mean
    denominator = ((n_d * (n - n_d) * variance) / (n - 1)) ** 0.5
    zG_star = numerator / denominator
    zG_star_abs = np.abs(zG_star)
    zScores.append(zG_star_abs)

#------------------------------------------------------------------------------
# 5. EXTRACT SMALLEST DISTANCE WITH SIGNIFICANT Z SCORE
#------------------------------------------------------------------------------

# Create 3D array by stacking the smooted data and significance scores for
# each distance
stackedValues = np.dstack(meanSmoothed)
zScoreValues = np.dstack(zScores)

# Choose a significance value
significance = 3
sigZScores = zScoreValues >= significance

# Get the index of the first true value representing first non random window size
index = np.argmax(sigZScores == True, axis=2)

# Need to identify those cells with no significant values in order to set the
# index value to the maximum window size
# Start by getting maximum significance value
maxSig = np.max(zScoreValues, axis=2)
# Identify those cells no significance values >= significance threshold
nonSig = maxSig < significance
# Place maximum index value for those cells with no signifcant values
np.place(index, nonSig==True, maxDistance - minDistance)

# Create row column values as 2D arrays
col, row = np.meshgrid(range(nRow), range(nCol))
# Extract the smoothed strike value and significance value
finalValues = stackedValues[row, col, index]
sigchosen = zScoreValues[row, col, index]

print("Maximum distance = " + str(np.nanmax(index) + minDistance))

#------------------------------------------------------------------------------
# 6. PLOT FIGURE
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
# Part (a) Plot original surface
#------------------------------------------------------------------------------

plt.subplot(1,3,1)
plt.xticks(np.arange(0))
plt.yticks(np.arange(0))

plt.contourf(lightningArray, levels=[0,0.2,0.4,0.6,0.8,1], cmap=plt.cm.RdPu_r, 
             extent=[minX,maxX,minY,maxY], origin='image')
plt.axis('off')

cbar = plt.colorbar(shrink=1, pad=0.05, orientation='horizontal')
cbar.outline.remove() 
cbar.ax.tick_params(direction='out')
cbar.ax.set_xlabel(r"Strike density ($\mathregular{km^{-2}}$)")

# Add area of interest rectangle
ax = fig.gca()
ax.set_aspect('equal') # to keep the array square

ax.text(0, 1, '(a)',
        horizontalalignment='right',
        verticalalignment='bottom',
        transform=ax.transAxes)
        
r = patches.Rectangle((1370000, 5145000), 30000, 30000, linewidth=1, fill=False, linestyle='solid', color='k')
ax.add_artist(r)
textLabel = r"Mean = " + str(round(mean, 3)) + "\nVariance = " + str(round(variance, 3))
ax.text(1370000 + 15000, 5145000 + 30000, textLabel, fontsize=6, style='italic',  
        verticalalignment='bottom', horizontalalignment='center')
        
#------------------------------------------------------------------------------
# Part (b) Plot significant distance
#------------------------------------------------------------------------------

plt.subplot(1,3,2)
plt.xticks(np.arange(0))
plt.yticks(np.arange(0))

# Plot significantdisatnce
plt.contourf(index + minDistance, levels=[5,9,13,17,21,25], cmap=plt.cm.YlOrRd, 
             extent=[minX,maxX,minY,maxY], origin='image')
#plt.imshow(index + minDistance, cmap=plt.cm.YlOrRd)


plt.axis('off')

cbar = plt.colorbar(shrink=1, pad=0.05, orientation='horizontal')
cbar.outline.remove() 
cbar.ax.tick_params(direction='out')
cbar.ax.set_xlabel(r"Minimum significant distance (cells)")

# Add area of interest rectangle
ax = fig.gca()
ax.set_aspect('equal') # to keep the array square

ax.text(0, 1, '(b)',
        horizontalalignment='right',
        verticalalignment='bottom',
        transform=ax.transAxes)
        
r = patches.Rectangle((1370000, 5145000), 30000, 30000, linewidth=1, fill=False, linestyle='solid', color='k')
ax.add_artist(r)
            
#------------------------------------------------------------------------------
# Part (c) Plot smoothed surface
#------------------------------------------------------------------------------

plt.subplot(1,3,3)
plt.xticks(np.arange(0))
plt.yticks(np.arange(0))

plt.contourf(finalValues, levels=[0,0.2,0.4,0.6,0.8,1], cmap=plt.cm.RdPu_r, 
             extent=[minX,maxX,minY,maxY], origin='image')
plt.axis('off')

cbar = plt.colorbar(shrink=1, pad=0.05, orientation='horizontal')
cbar.outline.remove() 
cbar.ax.tick_params(direction='out')
cbar.ax.set_xlabel(r"Smoothed strike density ($\mathregular{km^{-2}}$)")

# Add area of interest rectangle
ax = fig.gca()
ax.set_aspect('equal') # to keep the array square

ax.text(0, 1, '(c)',
        horizontalalignment='right',
        verticalalignment='bottom',
        transform=ax.transAxes)
        
r = patches.Rectangle((1370000, 5145000), 30000, 30000, linewidth=1, fill=False, linestyle='solid', color='k')
ax.add_artist(r)
    
#------------------------------------------------------------------------------

fig.savefig('figure4.png', dpi=150, bbox_inches='tight', pad_inches = 0.1)

#------------------------------------------------------------------------------


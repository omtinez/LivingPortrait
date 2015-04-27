
# coding: utf-8

# In[1]:

#from IPython.display import display, Image
#Image('pipeline.png')


# ### Shortest Path Between Frames
# $$S.S.D.: \quad D_{ij}=\sum(F_i-F_j)^2 \quad \forall \: i,j$$

# In[2]:

def imread(impath):
    ''' Utility function used to read an image into a 2-D array '''
    color = ndimage.imread(impath, mode='RGB')
    r, g, b = color[:,:,0], color[:,:,1], color[:,:,2]
    return (0.2989 * r + 0.5870 * g + 0.1140 * b).astype(np.float)

def videoVolume(imfolder):
    ''' Utility function used to read a series of images into a 3-D array '''
    imfiles = [f for f in listdir(imfolder) if isfile(join(imfolder,f)) and re.match('.*\.(png)|(jpg)', f.lower())]
    return [imread(join(imfolder,im)) for im in imfiles]

def SSD(video_volume):
    ''' Compute the sum of squared differences between all pairs of frames in a video volume '''
    output = np.zeros((len(video_volume), len(video_volume)), dtype=np.float)
    for i in range(len(video_volume)):
        curr_frame = video_volume[i].astype(np.float)
        for j in range(len(video_volume)):
            if i < j:
                output[i,j] = ((curr_frame - video_volume[j]) ** 2).sum()
            elif i > j:
                output[i,j] = output[j,i]
    return output

# In[3]:

def bestTransitions(ssd_matrix, transitions=100):
    ''' Find the subset of smallest transitions between frames '''
    L = ssd_matrix.shape[0]
    
    # Best transitions from i to j
    ij = np.array([np.argpartition(l, transitions)[:transitions] for l in ssd_matrix])
    # Best transitions from j to i (not strictly necessary due to simmetry)
    ji = np.array([[i for i in range(L) if j in ij[i]] for j in range(L)])
    
    # Build dependency maps to recover original edges after sampling
    graphix = np.array([np.array(['%d_%d' % (i,j) for j in ij[i]]) for i in range(L)]).flatten()
    graphmap = {graphix[x]:x for x in range(len(graphix))}
    
    return ij, ji, graphix, graphmap

# Solve:
#
# $$\sum_{ij \in A} w_{ij} x_{ij}$$
# 
# Subject to the following constraints:
# 
# $$x_{ij} \ge 0 \quad \forall \: ij \in A$$
# 
# $$\sum_j x_{ij} - \sum_j x_{ji} = \begin{cases}1, &\text{if }i=s;\\ -1, &\text{if }i=t;\\ 0, &\text{ otherwise.}\end{cases} \quad \forall \: i$$

# In[4]:

def linprogParams(ssd_matrix, ij, ji, graphix, graphmap, s, t):
    ''' Compute the linear programming parameters for the given problem '''
    L = ssd_matrix.shape[0]
    
    # 2-D array which, when matrix-multiplied by x, gives the values 
    # of the equality constraints at x
    A = np.zeros((L,len(graphmap)), dtype=np.int)
    for n in range(L):
        for i in ij[n]: A[n,graphmap['%d_%d' % (n,i)]] += 1
        for j in ji[n]: A[n,graphmap['%d_%d' % (j,n)]] -= 1
            
    # Coefficients of the linear objective function to be minimized
    c = np.zeros((len(graphix),))
    for n in range(len(graphix)):
        x,y = graphix[n].split('_')
        x = int(x)
        y = int(y)
        c[n] = ssd_matrix[x,y]
        
    # 1-D array of values representing the RHS of each equality 
    # constraint (row) in A
    b = np.zeros((L,))
    b[s] = 1
    b[t] = -1
    
    return c, A, b


# ### Frame Blending and Insertion
# 
# $$\alpha_i = i / n, \quad i \in 1..n$$
# 
# $$T_i = \alpha_i \cdot f(F_a) + (1-\alpha_i) \cdot f(F_b)$$
# 
# With *f(x)* being a simple Gaussian kernel

# In[5]:

def transitionFrames(frame1, frame2, numFrames=5):
    ''' Compute transition frames between two given frames '''
    blur1 = gaussian_filter(frame1, sigma=(.1,.1,0), order=0).astype(np.float)
    blur2 = gaussian_filter(frame2, sigma=(.1,.1,0), order=0).astype(np.float)
    blend = lambda alpha: (blur1*alpha + blur2*(1-alpha)).astype(np.uint8)
    return [blend(alpha) for alpha in np.linspace(1,0,numFrames)]
        
def easeTransitions(frameList, max_insertions=5):
    ''' Smooth out the worst transitions between consecutive frames '''
    
    for i in range(max_insertions):
        
        # Compute frame differences
        deltas = np.array([((frameList[i] - frameList[i+1]) ** 2).sum() for i in range(len(frameList)-1)])
        frameIndex = np.argmax(deltas) + 1

        # Done if no outliers are found
        if deltas.std() < deltas.mean(): break
            
        # Insert transition frames
        midFrames = transitionFrames(frameList[frameIndex-1], frameList[frameIndex])
        frameList = frameList[:frameIndex] + midFrames + frameList[frameIndex:]
        
    return frameList


# ### Key Frames and Actions

# In[6]:

def smoothSequence(videoName, startFrame, endFrame, action=0, transitions=100, ssd_matrix=None):
    ''' Find the smoothest and shortest possible sequence between two frames '''
    
    print('Computing difference matrix...')
    if ssd_matrix is None: ssd_matrix = SSD(videoVolume(videoname))
    
    print('Selecting %d best transitions for each frame...' % transitions)
    ij, ji, graphix, graphmap = bestTransitions(ssd_matrix, transitions=transitions)
    
    print('Populating linear programming parameters...')
    c, A, b = linprogParams(ssd_matrix, ij, ji, graphix, graphmap, startFrame, endFrame)
    
    print('Solving linear equations...')
    res = linprog(c, A_eq=A, b_eq=b, options={"disp": True})
    
    print('Ordering the transitions between frames...')
    unordered_transitions = [graphix[i] for i in range(len(res.x)) if res.x[i] > 0]
    nextFrame = lambda x: [int(t.split('_')[1]) for t in unordered_transitions if t.startswith('%d_' % x)][0]
    frameList = [startFrame]
    for i in range(len(unordered_transitions)):
        frameList += [nextFrame(frameList[-1])]
        
    print('Loading %d frames from input video...' % len(frameList))
    candle = lambda x: 'frame%04d.png' % (1+int(x))
    imlist = [join(videoname,candle(f)) for f in frameList]
    images = [ndimage.imread(img, mode='RGB') for img in imlist]
    
    print('Smoothing transitions between frames...')
    images = easeTransitions(images)
    
    print('Writing %d ordered frames to output location...' % len(images))
    fname, ext = splitext(imlist[0])
    for i,img in enumerate(images):
        toimage(img, cmin=0.0, cmax=...).save(join('out','A%03d_F%03d%s' % (action,i,ext)))
        
    # Return the number of frames output
    return len(images)

def keyFrameSequence(videoName, framelist, avoidFrames=[], loop=True, transitions=100):
    ''' Find the smoothest and shortest possible sequence containing a set of key frames '''
    
    vidLen = 0
    action = 0
    
    # Compute the diff matrix
    ssd_matrix = SSD(videoVolume(videoname))
    
    # Add the first frame at the end if looping around
    if loop: framelist += [framelist[0]]
    for i in range(len(framelist)-1):
        
        # Make a copy of the diff matrix and penalize other key frames
        diffMatrix = ssd_matrix.copy()
        for frame in framelist+avoidFrames:
            if frame == framelist[i] or frame == framelist[i+1]: continue
            diffMatrix[frame-12:frame+12] = diffMatrix.max() * 2
            diffMatrix.T[frame-12:frame+12] = diffMatrix.max() * 2
            
        # Find the sequence of frames between key frmaes
        vidLen += smoothSequence(videoName, framelist[i], framelist[i+1], 
                              action=action, transitions=transitions, ssd_matrix=diffMatrix)
        action += 1
    return vidLen


# ### Interactive Visualization

# In[7]:

#Image('ui.png', height=400)


# ### Other Applications
# 
# Combined with other computational photography and computer vision techniques, a number of other applications become available based on this work. For example, adding automatic key frame detection would make an interesting video summarization implementation. With the help of action recognition, the implementation described here can also be used in the process of extracting those actions from raw video. Also, even though computer generated graphics are nowadays virtually indistinguishable form actual footage, the concept of interactive videos can be used in the context of user interfaces such as digital assistants.
# 
# 
# ### Sample Implementation
# 
# The code below takes a folder containing all the frames from a portrait video as input, and it outputs all the frame sequences for all the actions occurring at the specified key frames. The resulting frames are then presented in an interactive application ready to use in most updated desktop web browsers at http://omtinez.com/portrait/.

# In[8]:

import numpy as np
from scipy import ndimage
from scipy.misc import toimage
from scipy.optimize import linprog 
from scipy.ndimage.filters import gaussian_filter

import re
import shutil
from os import listdir, unlink
from os.path import isfile, join, splitext

def delOutput():
    for f in [f for f in listdir('out') if isfile(join('out',f))]: unlink(join('out',f))


# In[9]:

delOutput()
videoname = 'face2'
vidLen = keyFrameSequence(videoname, [0,38,0,178,216,0,109,136])


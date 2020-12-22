#KHU-CVProject_01-Improved_Stereo_Matching
#Kwangwon Lee, 2016104142
#Also in Github: https://github.com/ryanleek/CV_Stereo_Matching

import numpy as np
import matplotlib.pyplot as plt
import os
from cv2 import cv2

#Name of folder where left and right img exists(code and folder in same dir.)
DATADIR = "sample"

#Read left and right image in grayscale with cv2 
#[Modified from self-project tutorial_03(https://github.com/ryanleek/TensorFlow)]
L_array = cv2.imread(os.path.join(DATADIR, "left.png"), cv2.IMREAD_GRAYSCALE)
R_array = cv2.imread(os.path.join(DATADIR, "right.png"), cv2.IMREAD_GRAYSCALE)

#Optional code to resize image, shape ratio must be changed for dif. img
#new_L_array = cv2.resize(L_array, (270,225))
#new_R_array = cv2.resize(R_array, (270,225))


#Height and Width of images
H, W = L_array.shape
#Array to store result of Stereo Matching
Result = np.zeros((H,W))
#Occlusion and Patch Size Value(Constant)
OCCLUSION = 1500
PATCHSIZE = 7

#New Height and Width of DSI, Cost, and Path
NEW_W = W-PATCHSIZE+1   #number of patches in a scanline(x-axis)
NEW_H = H-PATCHSIZE+1   #number of patches in y-axis
SHAPE = (NEW_W+1, NEW_W+1)  #Shape of DSI, Cost, and Path


#Stereo Matching with Disparity Space Image of each scanline
for k in range(0, NEW_H):

    DSI = np.zeros(SHAPE)

    #Gain DSI for each scanline of left image [Modified from Stereo Matching Sample code]
    for i in range(0, NEW_W):

        L_patch = L_array[k:k+PATCHSIZE-1, i:i+PATCHSIZE-1] #left patch

        for j in range(0, NEW_W):

            R_patch = R_array[k:k+PATCHSIZE-1, j:j+PATCHSIZE-1]  #right patch
            DSI[i+1,j+1] = np.sum((L_patch-R_patch)**2)  #use SSD for DSI value

    #Optional code to check DSI
    #plt.imshow(DSI, cmap="gray")
    #plt.show()
    

    #Array to store Cost and Path for DP 
    Cost = np.zeros(SHAPE)
    Path = np.zeros(SHAPE)

    #Set Boundary Condition
    for i in range (1, NEW_W+1):

        Cost[i,0] = OCCLUSION * i
        Cost[0,i] = OCCLUSION * i
        Path[i,0] = 2
        Path[0,i] = 3

    #Calculate Optimal Path using DP [From Lecture 8's pseudo code]
    for i in range(1, NEW_W+1):
        for j in range(1, NEW_W+1):

            #Array that stores cost of 3 paths
            costs = np.array([Cost[i-1,j-1]+DSI[i,j], Cost[i-1,j]+OCCLUSION, Cost[i,j-1]+OCCLUSION])

            Cost[i,j] = np.min(costs)   #store optimum path's cost
            Path[i,j] = np.argmin(costs) + 1    #store optimum path

    #Optional code to check Path Image
    #plt.imshow(Path, cmap="gray")
    #plt.show()


    #Set cursor to point current postion on DSI Path
    p, q = NEW_W, NEW_W #starts from lower right corner, p: y-axis val., q: x-axis val.

    #Recover Optimal Path to get Results [From Lecture 8's pseudo code]
    while p*q:  #while p, q are both not '0'

        if Path[p,q] == 1:      #p matches q, go diagonally
            
            Result[k,p] = abs(q-p)  #disparity score
           
            p = p-1
            q = q-1
        
        elif Path[p,q] == 2:    #unmatched, go up
           
            p = p-1
       
        elif Path[p,q] == 3:    #unmatched, go left
           
            q = q-1
    
    #Check if code is running [From Stereo Matching Sample code]
    print("!", end="", flush=True) 


#Check Result of Stereo Matching
#plt.imshow(Result, cmap="gray")
#plt.show()


#Improve Result by Occlusion Filling
for i in range(0, H):
    for j in range(1, W):
        if Result[i,j] == 0:    #if occulded pixel, copy from left pixel
            Result[i,j] = Result[i,j-1]

#Check final Result after Occlusion Filling
plt.imshow(Result, cmap="gray")
plt.show()
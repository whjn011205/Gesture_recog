""" Define functions to extract feature from raw gesture image data"""

import os
import cv2
import numpy as np
from numba import jit


## Detect hand patch of original image
def patchDetection(img):
    if img is None:
        return None
        
    blurred = cv2.GaussianBlur(img, (5, 5), 0) # Remove noise
    ret, otsu = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #contours, heirarchy = cv2.findContours(otsu, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    output = otsu
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if output[i,j]<255:
                output[i,j] = img[i,j]
    return output

## Crop the hand patch from the original image
def crop(path,sub,gesture,title):
    pixels = cv2.imread(path+sub+'//'+gesture+'//'+title+".jpg",cv2.IMREAD_GRAYSCALE)
    # crop the hand area
    framefile=open(path+sub+'//'+gesture+'//'+title+'.txt','r')
    handframe = framefile.read().split(' ')
    handframe = [int(x) for x in handframe]
    
    for i in handframe:
        if i <=0:
            #print("bad image:", path+sub+'//'+gesture+'//'+title+".jpg")
            return None
            
    framefile.close()
    rowstart = max(0,handframe[1]+2)
    rowend = min(pixels.shape[0],(handframe[1]+handframe[3]-2))
    colstart = max(0,(handframe[0]+2))
    colend = min(pixels.shape[1],(handframe[0]+handframe[2]-2))
    pixelsCropped =pixels[rowstart:rowend,colstart:colend]
    return pixelsCropped

## Get feature data for a single pixel in the image
@jit(nopython=True)
def featureExtractor(img,r,c,u0,v0):
    """
    given a pixel location and its deep value, return the depth feature for that pixel
    the formula is given by this paper: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/BodyPartRecognition.pdf
    :param x: 2D pixel location. its a tuple
    :param img: map a location to a depth value
    :param u: 2D offset1. its a tuple
    :param v: 2D offset2. its a tuple
    :return: depth feature for x
    """
    rowMax = img.shape[0]
    colMax = img.shape[1]
    
    depth =img[r,c]
    if depth==0:
        depth = 1
    u0_norm = int(round(u0[0]/depth))
    u1_norm = int(round(u0[1]/depth))
    v0_norm = int(round(v0[0]/depth))
    v1_norm = int(round(v0[1]/depth))
    
    ## Formulas to generate feature from crop hand patch
    pos1_r = r+u0_norm
    pos1_r = pos1_r % rowMax
    pos1_c = c+u1_norm
    pos1_c = pos1_c % colMax

    pos2_r = r+v0_norm
    pos2_r = pos2_r % rowMax
    pos2_c = c+v1_norm
    pos2_c = pos2_c % colMax

    a=np.int16(img[(pos1_r,pos1_c)])
    b=np.int16(img[(pos2_r,pos2_c)])

    return a-b

## Genereate feature data for the entire image
@jit
def getFeatures(path,sub, title,gesture,u,v):
    img = crop(path,sub,gesture,title)
    handPatch = patchDetection(img)
    if handPatch is None:
        return None
        
    L=handPatch.shape[0]
    H=handPatch.shape[1]
    #cv2.imshow(sub+","+gesture+","+title,handPatch)
    #cv2.waitKey(200)
    #cv2.destroyAllWindows()

    # column of output indicate the choice if u and v, 1:10
    # row of output indicate the pixel index. if pixel index is [x, y], if goes to x+y*ymax row of output
    #Sample 65x65 points from handpatch    
    output =np.zeros((65*65,10),dtype='int16')
    r_list=np.zeros((65),dtype='int16')    
    c_list=np.zeros((65),dtype='int16')
    for i in range(65):
        r_list[i]=int(i*L/65)
        c_list[i]=int(i*H/65)
    
    for i in range(10):
        u0 = (u[i*2],u[i*2+1])
        v0 = (v[i*2],v[i*2+1])

        for j,r in enumerate(r_list):
            #r = int(j*L/50)
            for k,c in enumerate(c_list):
                #c = int(k*H/50)
                #print(r,c)
                output[i,j+k*65] = featureExtractor(handPatch,r,c,u0,v0)
                
    return output
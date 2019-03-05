#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 10:42:52 2019

@author: ron
"""

import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import numpy as np
import pandas as pd
import cv2
import os
import glob

import matplotlib.pyplot as plt
import pickle



def get_hist(img):
    #input: binary image
    #output: histogram with respect to x-axis, but we are only 
    #interested in bottom half to set a starting point
    hist = np.sum(img[img.shape[0]//2:,:],axis=0)
    return hist
    

#input: warped filtered image
#output: arrays of pixels belong to left lane and right lane
def SlidingWindow(self, img, nWindows=10, margin=100, minP=10 ):
    hist = get_hist(img)
    leftstart = np.argmax(hist[ : np.int(hist.shape[0] // 2)])
    rightstart = np.argmax(hist[np.int(hist.shape[0] // 2) : ])
    opImg = np.dstack((BiWarped_img,BiWarped_img,BiWarped_img))*255
    
    hWindows = np.int(img.shape[0] / nWindows)
    
    #nonzero pixels in input image
    nonzeroP = np.nonzero(img)
    nzPy = np.array(nonzeroP[0])
    nzPx = np.array(nonzeroP[1])
    
    left_lane_ind=[]
    right_lane_ind=[]
    for win in range(nWindows):
        #current window boundary
        #because in cv, image y axis is 0 to large from top to bottom
        YLow = img.shape[0] - (win + 1) *hWindows
        YHigh = img.shape[0] - win *hWindows
        leftXL = leftstart - margin
        leftXR = leftstart + margin
        rightXL = rightstart - margin
        rightXR = rightstart + margin
        
        #if the window touches the border(may happen under big curve)
        #terminate the loop
        if leftXR >= img.shape[1] or leftXR <= 0 or rightXL >= img.shape[1] or rightXL <= 0:
            break
        
        #detect pixels that are located in current window
        #store their index in array nonzeroP
        #?
        pInWinL = np.where((nzPx > leftXL) & (nzPx < leftXR) & (nzPy>YLow) & (nzPy<YHigh))
        pInWinR = np.where((nzPx > rightXL) & (nzPx < rightXR) & (nzPy>YLow) & (nzPy < YHigh))
        
        left_lane_ind.append(pInWinL)
        right_lane_ind.append(pInWinR)
        
        if len(left_lane_ind) > minP:
            leftstart = np.int(np.mean(nzPx[pInWinL]))
        else:
            leftstart = leftstart + margin
        if len(right_lane_ind) > minP:
            rightstart = np.int(np.mean(nzPx[pInWinR]))
        else:
            leftstart = leftstart + margin
        
        #draw a green square on image
        cv2.Rectangle(opImg, [leftXL, YHigh],[leftXR, YLow],[0, 255, 0], 1)
        cv2.Rectangle(opImg, [rightXL, YHigh],[rightXR, YLow],[0, 255, 0], 1)
        
    left_lane_ind = np.concatenate(left_lane_ind)
    right_lane_ind = np.concatenate(right_lane_ind)
    
    leftX = nzPx[left_lane_ind]
    leftY = nzPy[left_lane_ind]
    rightX = nzPx[right_lane_ind]
    rightY = nzPy[right_lane_ind]

    return opImg, leftX, leftY, rightX, rightY, left_lane_ind, right_lane_ind
    
#input: last argument will be image result from slidingWindow()
def PolyFit(self, leftX, leftY, rightX, rightY, left_lane_ind, right_lane_ind, BiWarped_img):
    
    polyParaL = np.polyfit(leftY, leftX, 2)
    polyParaR = np.polyfit(rightY, rightX, 2)
    yrange = np.linspace(0, img.shape[0]-1, img.shape[0])
    polyL = polyParaL[0] * yrange ^ 2 + polyParaL[1] * yrange + polyParaL
    polyR = polyParaR[0] * yrange ^ 2 + polyParaR[1] * yrange + polyParaR
    
    nonzero = np.nonzero(BiWarped_img)
    nzPY = np.array(nonzero[0])
    nzPX = np.array(nonzero[1])
    opImg = np.dstack((BiWarped_img,BiWarped_img,BiWarped_img))*255
    opImg[nzPY[left_lane_ind], nzPX[left_lane_ind]]=(255,0,0)
    opImg[nzPY[right_lane_ind], nzPX[right_lane_ind]]=(0,0,255)
    
    return opImg, polyL, polyR

#input: original image
def DrawDetectedLane(self, polyL, polyR, img):
    LaneMask = np.zeros_like(img)
    yrange = np.linspace(0, img.shape[0]-1, img.shape[0])
    
    leftPoints = np.array(np.transpose(np.vstack((polyL,yrange))))
    rightPoints = np.array(np.flipud(np.transpose(np.vstack((polyR,yrange)))))
    PolyPoints = np.hstack((leftPoints, rightPoints))
    
    cv2.fillPoly(LaneMask, PolyPoints, (84, 113, 170))
    inv = inv_PT(LaneMask)
    inv = cv2.addWeighted(img, 1, inv, 0,7,0)
    return inv
    
    
























































































































































































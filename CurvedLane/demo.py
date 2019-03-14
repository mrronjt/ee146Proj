#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 21:57:14 2019

@author: ron
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import numpy as np

import cv2
import os
import glob
import matplotlib.pyplot as plt
import pickle
from moviepy.editor import VideoFileClip
import math

counter = 0

class lane_detection():
    
    
    def undistort_img(self):
        # Prepare object points 0,0,0 ... 8,5,0
        obj_pts = np.zeros((6*9,3), np.float32)
        obj_pts[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)
    
        # Stores all object points & img points from all images
        objpoints = []
        imgpoints = []
    
        # Get directory for all calibration images
        images = glob.glob('camera_cal/*.jpg')
    
        for indx, fname in enumerate(images):
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
            ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
    
            if ret == True:
                objpoints.append(obj_pts)
                imgpoints.append(corners)
        # Test undistortion on img
        img_size = (img.shape[1], img.shape[0])
    
        # Calibrate camera
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None,None)
    
        dst = cv2.undistort(img, mtx, dist, None, mtx)
        # Save camera calibration for later use
        dist_pickle = {}
        dist_pickle['mtx'] = mtx
        dist_pickle['dist'] = dist
        pickle.dump( dist_pickle, open('camera_cal/cal_pickle.p', 'wb') )
    
    def undistort(self, img, cal_dir='camera_cal/cal_pickle.p'):
        
        with open(cal_dir, mode='rb') as f:
            file = pickle.load(f)
        mtx = file['mtx']
        dist = file['dist']
        dst = cv2.undistort(img, mtx, dist, None, mtx)
        
        return dst
    
    def preprocess(self, img, s_thresh=(100, 255), sx_thresh=(15, 255)):
        img = self.undistort(img)
        img = np.copy(img)
        # Convert to HLS color space and separate the V channel
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
        l_channel = hls[:,:,1]
        s_channel = hls[:,:,2]
        h_channel = hls[:,:,0]
        # Sobel x
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 1) # Take the derivative in x
        abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
        scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))        
        # Threshold x gradient
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1        
        # Threshold color channel
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
        combined_binary = np.zeros_like(sxbinary)
        combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
        return combined_binary
    
    def perspective_warp(self, img):
        src = self.src
        dst = self.dst
        dst_size=(img.shape[1],img.shape[0])
        img_size = np.float32([(img.shape[1],img.shape[0])])
        src = src* img_size
        # For destination points, I'm arbitrarily choosing some points to be
        # a nice fit for displaying our warped result 
        # again, not exact, but close enough for our purposes
        dst = dst * np.float32(dst_size)
        # Given src and dst points, calculate the perspective transform matrix
        M = cv2.getPerspectiveTransform(src, dst)
        # Warp the image using OpenCV warpPerspective()
        warped = cv2.warpPerspective(img, M, dst_size)
        return warped
    
    def inv_PT(self, img):
        dst = self.src
        src = self.dst
        dst_size=(img.shape[1],img.shape[0])
        img_size = np.float32([(img.shape[1],img.shape[0])])
        src = src* img_size
        dst = dst * np.float32(dst_size)
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(img, M, dst_size)
        return warped
    
    
    #input: binary image
    #output: histogram with respect to x-axis, but we are only 
    #interested in bottom half to set a starting point
    def get_hist(self, img):
        hist = np.sum(img[img.shape[0]*2//3:,:],axis=0)
        return hist
        
    
    
    
    #input: warped filtered image
    #output: arrays of pixels belong to left lane and right lane
    def SlidingWindow(self, BiWarped_img, nWindows=10, margin=50, minP=1 ):
        hist = self.get_hist(BiWarped_img)
        leftstart = np.argmax(hist[ : np.int(hist.shape[0] // 2)]) #index(x value) of max pixel in left of histogram
        rightstart = np.argmax(hist[np.int(hist.shape[0] // 2) : ]) + np.int(hist.shape[0] // 2) #index of max pixel in right of histogram
        opImg = np.dstack((BiWarped_img,BiWarped_img,BiWarped_img))*255
        hWindows = np.int(BiWarped_img.shape[0] / nWindows)
        
        #nonzero pixels in input image
        nonzeroP = np.nonzero(BiWarped_img)#returns the indices of non-zero pixels
        #nzPy = np.array(nonzeroP[0]) #indices of y value
        #nzPx = np.array(nonzeroP[1]) #indices of x value
        
        left_lane_ind=[]
        right_lane_ind=[]
        slideRightX=[]
        slideLeftX=[]
        slideY=[]
        
        for win in range(nWindows):
            #current window boundary
            #because in cv, image y axis is 0 to large from top to bottom
            YLow = BiWarped_img.shape[0] - (win + 1) *hWindows #coor
            YHigh = BiWarped_img.shape[0] - win *hWindows #coor
            Ymid = (YLow + YHigh) / 2
            slideY.append(Ymid)
            leftXL = leftstart - margin #index
            leftXR = leftstart + margin #index 
            rightXL = rightstart - margin #index
            rightXR = rightstart + margin #index
            slideLeftX.append(leftstart)
            slideRightX.append(rightstart)
            #detect pixels that are located in current window
            #store their index in array nonzeroP
            pInWinL = ((nonzeroP[0] >= YLow) & (nonzeroP[0] < YHigh) & (nonzeroP[1] >= leftXL) &  (nonzeroP[1] < leftXR)).nonzero()[0]
            pInWinR = ((nonzeroP[0] >= YLow) & (nonzeroP[0] < YHigh) & (nonzeroP[1] >= rightXL) &  (nonzeroP[1] < rightXR)).nonzero()[0]
            left_lane_ind.append(pInWinL)
            right_lane_ind.append(pInWinR)
            
            
            if len(pInWinL) > minP:
                leftstart = np.int(np.mean(nonzeroP[1][pInWinL]))
            else:
                leftstart = leftstart 
            if len(pInWinR) > minP:
                rightstart = np.int(np.mean(nonzeroP[1][pInWinR]))
            else:
                rightstart = rightstart 
            
            cv2.rectangle(opImg, (leftXL, YHigh),(leftXR, YLow),[255, 0, 0], 1)
            cv2.rectangle(opImg, (rightXL, YHigh),(rightXR, YLow),[0, 0, 255], 1)
            
        left_lane_ind = np.concatenate(left_lane_ind)
        right_lane_ind = np.concatenate(right_lane_ind)
        
        leftX = nonzeroP[1][left_lane_ind]
        leftY = nonzeroP[0][left_lane_ind]
        rightX = nonzeroP[1][right_lane_ind]
        rightY = nonzeroP[0][right_lane_ind]
        
    
        return opImg, (leftX, leftY), (rightX, rightY), (left_lane_ind, right_lane_ind), (slideLeftX, slideRightX, slideY)
        
    #input: last argument will be image result from slidingWindow()
    def PolyFit(self, left, right, left_lane_ind, right_lane_ind, BiWarped_img):        
        polyParaL = np.polyfit(left[1], left[0], 2)
        polyParaR = np.polyfit(right[1], right[0], 2)        
        yrange = np.linspace(0, BiWarped_img.shape[0]-1, BiWarped_img.shape[0])
        polyL = polyParaL[0] * yrange ** 2 + polyParaL[1] * yrange + polyParaL[2]
        polyR = polyParaR[0] * yrange ** 2 + polyParaR[1] * yrange + polyParaR[2]               
        nonzeroP = np.nonzero(BiWarped_img)
        opImg = np.dstack((BiWarped_img,BiWarped_img,BiWarped_img))*255
        opImg[nonzeroP[0][left_lane_ind], nonzeroP[1][left_lane_ind]]=(255,0,0)
        opImg[nonzeroP[0][right_lane_ind], nonzeroP[1][right_lane_ind]]=(0,0,255)
        
        return opImg, polyL, polyR
    
    #input: original image
    def DrawDetectedLane(self, polyL, polyR, img):
        LaneMask = np.zeros_like(img)
        plt.imshow
        yrange = np.linspace(0, img.shape[0]-1, img.shape[0])
        
        leftPoints = np.array([np.transpose(np.vstack([polyL,yrange]))])
        rightPoints = np.array([np.flipud(np.transpose(np.vstack([polyR,yrange])))])
        PolyPoints = np.hstack((leftPoints, rightPoints))
        cv2.fillPoly(LaneMask, np.int_(PolyPoints), (84, 113, 170))
        inv = self.inv_PT(LaneMask)
        inv = cv2.addWeighted(img, 1, inv, 0.9,0)
        return inv
        
    
     
    
    def readBenchMark(self, filename):
        flag = 0;
        with open(filename,"r") as fp:
            for i, line in enumerate(fp):
                if i == 2:
                    flag = 1;
                    third = line.split(' ') 
                if i == 1:
                    second = line.split(' ')
        if flag == 1:
            if (len(third) < 10 or len(second) < 10):
                return (0, 0),  (0,0), 0
            for i in range(len(third)-1):
                third[i]=float(third[i])#make a list of float
            for i in range(len(second)-1):
                second[i]=float(second[i])#make a list of float                    
            third = third[:-1]
            second = second[:-1]
            rightx=[]
            righty=[]
            leftx=[]
            lefty=[]
            for i in range(0,len(third)-1,2):
                rightx.append(third[i])
                righty.append(third[i+1])
            for i in range(0,len(second)-1,2):                
                leftx.append(second[i])
                lefty.append(second[i+1])                        
            
            iterL=0
            iterR=0
            increL=math.floor(len(leftx)/10)
            increR=math.floor(len(rightx)/10)
            leftxx=[];
            rightxx=[];
            leftyy=[];
            rightyy=[];
            print('leftx len is '+ str(len(leftx)))
            while len(leftxx) < 10:                
                if iterL >= len(leftx):
                    iterL = len(leftx) -1
                if iterR >= len(rightx):
                    iterR = len(rightx) -1
                leftxx.append(leftx[iterL])
                rightxx.append(rightx[iterR])
                leftyy.append(lefty[iterL])
                rightyy.append(righty[iterR])
                iterL = iterL + increL
                iterR = iterR + increR
        else:
            leftxx=0
            leftyy=0
            rightxx=0
            rightyy=0
        return (leftxx, leftyy),  (rightxx,rightyy), flag
    
    def calError(self, myleft, myright, slide, file):
        Bleft, Bright, flag = self.readBenchMark(file)
        if flag == 1:
            BleftX = np.asarray(Bleft[0],dtype=np.float32)
            BleftY = np.asarray(Bleft[1])
            BrightX = np.asarray(Bright[0])
            BrightY = np.asarray(Bright[1])
            slideleftX = np.asarray(slide[0])
            sliderightX = np.asarray(slide[1])
            slideY = np.array(slide[2])
            leftError = np.mean(np.sqrt((slideleftX-BleftX)**2 + (slideY-BleftY)**2))
            rightError = np.mean(np.sqrt((sliderightX-BrightX)**2 + (slideY-BrightY)**2))
        else:
            leftError = 0
            rightError = 0
        return leftError, rightError

    def __init__(self):
        
        self.img = cv2.imread('test_images/CU/CU1/01800.jpg', cv2.IMREAD_COLOR)        
        self.src = np.float32([(0.45,0.50),(0.49,0.50),(0.30,0.68),(0.60,0.68)]) #for CU 1640 x 590 0.73
        self.dst= np.float32([(0.4,0), (0.7, 0), (0,1), (0.9,1)])
        self.direc = "./test_images/CU/CU1"
#        self.direc = "./test_images/CU/CU4"
#        self.direc = "./test_images/CU/CU2"
#        self.direc = "./test_images/CU"
        self.file_list = sorted(glob.glob(os.path.join(self.direc,"*.txt")))

    
    
    def main(self):
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        plt.figure()
        plt.imshow(self.img)
        dst = self.preprocess(self.img)#threshold 
        dst = self.perspective_warp(dst)
        opImg, leftCoor, rightCoor, PixelOrder, slide = self.SlidingWindow(dst)
        plt.figure()
        plt.imshow(opImg)
        file = self.file_list[177]
        print("file list len is "+str(len(self.file_list)))
        print(file)
        if len(leftCoor)!= 0 and len(rightCoor)!= 0:
            leftError, rightError = self.calError(leftCoor, rightCoor, slide, file)
            opImg, polyL, polyR = self.PolyFit(leftCoor, rightCoor, PixelOrder[0], PixelOrder[1], dst)
            OverlayedImg = self.DrawDetectedLane(polyL, polyR, self.img)
            print("leftError is " +str(leftError))
            print("rightError is " +str(rightError))
        else:
            OverlayedImg = self.img
        plt.figure()
        plt.imshow(OverlayedImg)
        
    def forVid(self, img):
        global counter
        dst = self.preprocess(img)#threshold 
        dst = self.perspective_warp(dst)
        opImg, leftCoor, rightCoor, PixelOrder, slide = self.SlidingWindow(dst)
        if counter <= len(self.file_list)-1:
            file = self.file_list[counter]            
            if len(leftCoor[0])!= 0 and len(leftCoor[1])!= 0 and len(rightCoor[0])!= 0 and len(rightCoor[1])!= 0:
                leftError, rightError = self.calError(leftCoor, rightCoor, slide, file)
                opImg, polyL, polyR = self.PolyFit(leftCoor, rightCoor, PixelOrder[0], PixelOrder[1], dst)
                OverlayedImg = self.DrawDetectedLane(polyL, polyR, img)
                with open("ErrorOP_CU1.txt","a") as text_file:
                    text_file.write("%f %f " % (leftError, rightError))
            else:
                return img
        counter = counter + 1
                
        return OverlayedImg
        
if __name__ == '__main__':
    lane = lane_detection()
#    lane.main()
    myclip = VideoFileClip('test_images/CU/CU1/CU1_5fps.mp4')
#    myclip = VideoFileClip('test_images/CU/CU3_5fps.mp4')
#    myclip = VideoFileClip("challenge_video.mp4")
    output_vid = 'OP_CU1_5fps.mp4'
    clip = myclip.fl_image(lane.forVid)
    clip.write_videofile(output_vid, audio=False)
    


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 10:51:17 2019

@author: ron
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#import sys
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import numpy as np
#import pandas as pd
import cv2
#import os
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
        #cv2.imwrite('camera_cal/test_cal.jpg', dst)
        with open(cal_dir, mode='rb') as f:
            file = pickle.load(f)
        mtx = file['mtx']
        dist = file['dist']
        dst = cv2.undistort(img, mtx, dist, None, mtx)
        
        return dst
    
    def pipeline(self, img, s_thresh=(100, 255), sx_thresh=(15, 255)):
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
        
        color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
        
        combined_binary = np.zeros_like(sxbinary)
        combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
        return combined_binary

#    self.src = np.float32([(0.38,0.42),(0.56,0.42),(0.14,0.83),(0.90,0.83)]) #for cal
#    self.dst=np.float32([(0.05,0), (0.9, 0), (0.1,1), (0.9,1)])
    
    def perspective_warp(self, img#, #src=self.src, dst=self.dst
                         #dst_size=(img.shape[1],img.shape[0]),
                         #dst_size=(1280,720),
                         #src=np.float32([(0.43,0.65),(0.58,0.65),(0.1,1),(1,1)]),#for original test image 
                         #(550,302),(742,302),(128,720),(1280,720)
                         #src=np.float32([(0.45,0.51),(0.49,0.51),(0.36,0.76),(0.60,0.76)]), #for CULane
                         #(600,450),(1000,450),(750,250),(800,250) for 1640 x 590
                         #src=np.float32([(0.40,0.45),(0.48,0.45),(0.30,0.76),(0.66,0.76)]), #for CULane
                         #(700,250),(800,250),(500,450),(1050,450) for 1640 x 590
#                             src=np.float32([(0.38,0.42),(0.56,0.42),(0.14,0.83),(0.90,0.83)]), #for cal
                         #(250,350),(350,350),(170,450),(500,450) for 640 x 480
#                             dst=np.float32([(0,0), (1, 0), (0,1), (1,1)])
                         ):
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
    
    def inv_PT(self, img#, #src=self.dst, dst=self.src#dst_size=(img.shape[1],img.shape[0]),
#                         src=np.float32([(0.1,0), (0.9, 0), (0.1,1), (0.9,1)]),
#                             src=np.float32([(0,0), (1, 0), (0,1), (1,1)]),             
#                                        src=np.float32([(0.05,0), (0.9, 0), (0.1,1), (0.9,1)]),
#                         dst=np.float32([(0.43,0.65),(0.58,0.65),(0.1,1),(1,1)])#for original test image
                         #dst=np.float32([(0.45,0.51),(0.49,0.51),(0.36,0.76),(0.60,0.76)]), #for CULane
                         #dst=np.float32([(0.40,0.45),(0.48,0.45),(0.30,0.76),(0.66,0.76)]), #for CULane
                         #(700,250),(800,250),(500,450),(1050,450) for 1640 x 590
#                                         dst=np.float32([(0.38,0.42),(0.56,0.42),(0.14,0.83),(0.90,0.83)]) #for cal
                         #(250,350),(350,350),(170,450),(500,450) for 640 x 480 0.83
                         ):
        dst = self.src
        src = self.dst
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
#        print(rightstart)
        opImg = np.dstack((BiWarped_img,BiWarped_img,BiWarped_img))*255
        #print("leftstart is " + str(leftstart))
        #print("rightstart is " + str(rightstart))
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
#            print ("leftXL is " + str(leftXL))
#            #print ("leftXL coor is " + str(BiWarped_img[nonzeroP[1][leftXL], [nonzeroP[1][leftXR]]]))
#            print ("leftXR is " + str(leftXR))
#            #print ("leftXR coor is " + str(BiWarped_img[nonzeroP[1][leftXR]]))
#            print ("rightXL is " + str(rightXL))
#            #print ("rightXL coor is " + str(BiWarped_img[nonzeroP[1][rightXL]]))
#            print ("rightXR is " + str(rightXR))
#            #print ("rightXR coor is " + str(BiWarped_img[nonzeroP[1][rightXR]]))
#            
#            #print ("height is " + str(BiWarped_img.shape[0]))
#            print ("width is " + str(BiWarped_img.shape[1]))
#            #if the window touches the border(may happen under big curve)
            #terminate the loop
#            print(win)
            #if BiWarped_img[nonzeroP[1][leftXR]] >= self.img.shape[1] or BiWarped_img[nonzeroP[1][leftXL]] <= 0 or BiWarped_img[nonzeroP[1][rightXR]] >= self.img.shape[1] or BiWarped_img[nonzeroP[1][rightXL]] <= 0:
#            if leftXL <= 0 or rightXL <= 0 or leftXR >= BiWarped_img.shape[1]  or rightXR >= BiWarped_img.shape[1]:
#                break
                
#            print("Passed if statement")
            
            #detect pixels that are located in current window
            #store their index in array nonzeroP
            #?
            #pInWinL = np.nonzero((nonzeroP[1] > leftXL) & (nonzeroP[1] < leftXR) & (nonzeroP[0] > YLow) & (nonzeroP[0] < YHigh))[0]
            #pInWinR = np.nonzero((nonzeroP[1] > rightXL) & (nonzeroP[1] < rightXR) & (nonzeroP[0] > YLow) & (nonzeroP[0] < YHigh))[0]
            
            pInWinL = ((nonzeroP[0] >= YLow) & (nonzeroP[0] < YHigh) & (nonzeroP[1] >= leftXL) &  (nonzeroP[1] < leftXR)).nonzero()[0]
            pInWinR = ((nonzeroP[0] >= YLow) & (nonzeroP[0] < YHigh) & (nonzeroP[1] >= rightXL) &  (nonzeroP[1] < rightXR)).nonzero()[0]
            
            #pInWinL = ((nzPy >= YLow) & (nzPy < YHigh) & (nzPx >= leftXL) &  (nzPx < leftXR)).nonzero()[0]
            #pInWinR = ((nzPy >= YLow) & (nzPy < YHigh) & (nzPx >= rightXL) &  (nzPx < rightXR)).nonzero()[0]
#            print(win)
#            print("pInWinL size is "+ str(pInWinL.shape))
#           print("pinWinL is "+ str(pInWinL))
#            print("pinWinR is "+ str(pInWinR))
            left_lane_ind.append(pInWinL)
            right_lane_ind.append(pInWinR)
            
            
#            if len(left_lane_ind) > minP:
            if len(pInWinL) > minP:
                leftstart = np.int(np.mean(nonzeroP[1][pInWinL]))
                #leftstart = np.int(np.mean(nzPx[pInWinL]))
            else:
#                print('left + margin')
                leftstart = leftstart #+ margin
#            if len(right_lane_ind) > minP:
            if len(pInWinR) > minP:
                rightstart = np.int(np.mean(nonzeroP[1][pInWinR]))
                #rightstart = np.int(np.mean(nzPx[pInWinR]))
            else:
#                print('right + margin')
                rightstart = rightstart #- margin
            
            #draw a green square on image
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
        
#        if len(left)!=0 and len(right[0])!= 0 and len(right[1])!=0 :
#            print('not null')
#        print('left1 is '+str(left[1]))
#        print('left0 is '+str(left[0]))
#        print('right1 is '+str(right[1]))
#        print('right0 is '+str(right[0]))
        
        polyParaL = np.polyfit(left[1], left[0], 2)
        polyParaR = np.polyfit(right[1], right[0], 2)
        
#        left_a=[]
#        left_b=[]
#        left_c=[]
#        right_a=[]
#        right_b=[]
#        right_c=[]
#        left_a.append(polyParaL[0])
#        left_b.append(polyParaL[1])
#        left_c.append(polyParaL[2])
#        right_a.append(polyParaR[0])
#        right_b.append(polyParaR[1])
#        right_c.append(polyParaR[2])
#        left_fit_=np.empty(3)
#        right_fit_=np.empty(3)
#        left_fit_[0]=np.mean(left_a[-10:])
#        left_fit_[1]=np.mean(left_b[-10:])
#        left_fit_[2]=np.mean(left_c[-10:])
#        right_fit_[0]=np.mean(right_a[-10:])
#        right_fit_[1]=np.mean(right_b[-10:])
#        right_fit_[2]=np.mean(right_c[-10:])
        #print("left_fit_ is " + str(left_fit_))
        #print("right_fit_ is "+ str(right_fit_))
        #print(type(left_fit_[0]))
        
        yrange = np.linspace(0, BiWarped_img.shape[0]-1, BiWarped_img.shape[0])
#        print("polyParaL is " + str(polyParaL))
#        print("polyParaR is " + str(polyParaR))
#        print(yrange.size)
        polyL = polyParaL[0] * yrange ** 2 + polyParaL[1] * yrange + polyParaL[2]
        polyR = polyParaR[0] * yrange ** 2 + polyParaR[1] * yrange + polyParaR[2]
        #polyL = left_fit_[0] * yrange ** 2 + left_fit_[1] * yrange + left_fit_[2]
        #polyR = right_fit_[0] * yrange ** 2 + right_fit_[1] * yrange + right_fit_[2]
        
        
        nonzeroP = np.nonzero(BiWarped_img)
        #nzPY = np.array(nonzero[0])
        #nzPX = np.array(nonzero[1])
        opImg = np.dstack((BiWarped_img,BiWarped_img,BiWarped_img))*255
        opImg[nonzeroP[0][left_lane_ind], nonzeroP[1][left_lane_ind]]=(255,0,0)
        opImg[nonzeroP[0][right_lane_ind], nonzeroP[1][right_lane_ind]]=(0,0,255)
        
        return opImg, polyL, polyR
    
    #input: original image
    def DrawDetectedLane(self, polyL, polyR, img):
        LaneMask = np.zeros_like(img)
        plt.imshow
        yrange = np.linspace(0, img.shape[0]-1, img.shape[0])
        
#        print("yrange size is " + str(yrange.shape))
#        print("polyL size is " + str(polyL.shape))
#        #print(polyL)
        leftPoints = np.array([np.transpose(np.vstack([polyL,yrange]))])
#        print("leftpoint size is "+ str(leftPoints.shape))
        rightPoints = np.array([np.flipud(np.transpose(np.vstack([polyR,yrange])))])
#        print("rightpoint size is "+ str(rightPoints.shape))
        PolyPoints = np.hstack((leftPoints, rightPoints))
        #PolyPoints = np.int32(PolyPoints)
        #print("polyPoint type is " + str(type((PolyPoints[3]))))
        #PolyPoints = np.array([PolyPoints], dtype = np.int32)
#        print('PolyPoint is '+ str(PolyPoints.shape))
#        PolyPoints = PolyPoints.reshape(-1,1,2)
#        print('PolyPoint shape is '+ str(PolyPoints.shape))
        #cv2.fillPoly(LaneMask, np.array([PolyPoints], np.int32), (84, 113, 170))
        cv2.fillPoly(LaneMask, np.int_(PolyPoints), (84, 113, 170))
#        plt.figure()
#        plt.imshow(LaneMask)
        #cv2.fillPoly(LaneMask, np.array([PolyPoints], 'int32'), (84, 113, 170))
        #cv2.fillPoly(LaneMask, np.int32([PolyPoints]), (84, 113, 170))
        inv = self.inv_PT(LaneMask)
        #print('inv is here')
        #plt.imshow(inv)
        inv = cv2.addWeighted(img, 1, inv, 0.9,0)
        return inv
        
    
     
    
    def readBenchMark(self, filename):
        flag = 0;
#        with open("./test_images/CU/CU1/00000.lines.txt","r") as fp:
        with open(filename,"r") as fp:
#            print("opened")
#            print(fp)
            for i, line in enumerate(fp):
                
#                print("at loop "+ str(i))
                if i == 2:
#                    print("in third")
                    flag = 1;
                    third = line.split(' ') 
#                    print(third[0])
                if i == 1:
                    second = line.split(' ')
        #            print(third)
#        print(second)
#        print("here")
        if flag == 1:
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
            
            
            iter=0;
            leftxx=[];
            rightxx=[];
            leftyy=[];
            rightyy=[];
            print('leftx len is '+ str(len(leftx)))
            while len(leftxx) < 10:
    #            print("len is " + str(len(leftxx)))
    #            print(iter)
                if iter >= len(leftx):
                    iter = len(leftx) -1
                leftxx.append(leftx[iter])
                rightxx.append(rightx[iter])
                leftyy.append(lefty[iter])
                rightyy.append(righty[iter])
    #            print("len is " + str(len(leftxx)))
                iter = iter + 3
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
    #        print(BleftX)
    #        print(slideY.shape)
    #        print(BleftX.shape)
    #        print("err is "+str(Bleft[0][1]+slide[0][1]))
    #        print(type(slide[0]))
    #        leftError = np.mean(((slide[0]-BleftX)**2 + (slide[2]-BleftY)**2))
    #        rightError = np.mean(((slide[1]-BrightX)**2 + (slide[2]-BrightY)**2))
            leftError = np.mean(np.sqrt((slideleftX-BleftX)**2 + (slideY-BleftY)**2))
            rightError = np.mean(np.sqrt((sliderightX-BrightX)**2 + (slideY-BrightY)**2))
        else:
            leftError = 0
            rightError = 0
        return leftError, rightError

    
    
    def __init__(self):
        
        self.img = cv2.imread('test_images/CU/CU1/01800.jpg', cv2.IMREAD_COLOR)
#        self.img = cv2.imread('test_images/test1.jpg')
        
        self.src = np.float32([(0.45,0.50),(0.49,0.50),(0.30,0.68),(0.60,0.68)]) #for CU 1640 x 590 0.73
        self.dst= np.float32([(0.4,0), (0.7, 0), (0,1), (0.9,1)])
#        self.direc = "./test_images/CU/CU1"
#        self.direc = "./test_images/CU"
        self.direc = "./test_images/CU/CU2"
        self.file_list = sorted(glob.glob(os.path.join(direc,"*.txt")))

    
    
    def main(self):
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        plt.figure()
        plt.imshow(self.img)
        dst = self.pipeline(self.img)#threshold 
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
        
        print("mycounter: ", str(counter))
        dst = self.pipeline(img)#threshold 
        dst = self.perspective_warp(dst)
        opImg, leftCoor, rightCoor, PixelOrder, slide = self.SlidingWindow(dst)
        if counter <= len(self.file_list)-1:
            file = self.file_list[counter]
            if len(leftCoor[0])!= 0 and len(leftCoor[1])!= 0 and len(rightCoor[0])!= 0 and len(rightCoor[1])!= 0:
                leftError, rightError = self.calError(leftCoor, rightCoor, slide, file)
    #            print("length is "+ str(len(leftCoor)))
                opImg, polyL, polyR = self.PolyFit(leftCoor, rightCoor, PixelOrder[0], PixelOrder[1], dst)
                OverlayedImg = self.DrawDetectedLane(polyL, polyR, img)
                with open("ErrorOP.txt","a") as text_file:
                    text_file.write("%f %f " % (leftError, rightError))
            else:
                return img
        counter = counter + 1
        
        
        return OverlayedImg
        
        



  

if __name__ == '__main__':
    #img = cv2.imread('test_images/test1.jpg', cv2.IMREAD_COLOR)
    #cv2.imshow("image", img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    lane = lane_detection()
#    lane.main()
#    myclip = VideoFileClip('test_images/CU/CU1/CU1_5fps.mp4')
    myclip = VideoFileClip('test_images/CU/CU2/CU2_5fps.mp4')
     
#    myclip = VideoFileClip("challenge_video.mp4")
    output_vid = 'OP_CU3_5fps.mp4'
    clip = myclip.fl_image(lane.forVid)
    clip.write_videofile(output_vid, audio=False)
    

#img = cv2.imread('test_images/test1.jpg')#input image
        #input video

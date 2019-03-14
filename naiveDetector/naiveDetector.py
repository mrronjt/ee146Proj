#import sys
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import cv2 as cv
import numpy as np
import os
import glob
import math

# CONSTANTS
LOWERLINEBOUND = 150
ROITOP = 220
ROIMID = 750
ROILEFT = 450
ROIRIGHT = 1000
ROIBOTTOM = 425

def left_right_lines(frame, lines):
    left = []
    right = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        linear_values = np.polyfit((x1, x2), (y1,y2), 1)
        slope = linear_values[0]
        y_intercept = linear_values[1]

        if abs(slope) < 1e-1:
            continue

        if slope < 0:
            left.append((slope, y_intercept))
        else:
            right.append((slope, y_intercept))
    
    #print("right: ", str(right))
    if len(left) == 0:
        left.append((-1e-1,300))
    if len(right) == 0:
        right.append((1e-1,300))

    left_avg = np.average(left, axis=0)
    right_avg = np.average(right, axis=0)

    #print("left_avg: ", str(left_avg))
    #print("right_avg: ", str(right_avg))

    left_line = linecalc(frame, left_avg)
    right_line = linecalc(frame, right_avg)

    return np.array([left_line, right_line])

def linecalc(frame, linevalues):
    slope, intercept = linevalues

    ystart = ROIBOTTOM+30
    yend = int(ystart - LOWERLINEBOUND)
    xstart = int((ystart - intercept) / slope)
    xend = int((yend - intercept) / slope)

    return np.array([xstart, ystart, xend, yend])

def readBenchMark(filename):
        flag = 0
        with open(filename,"r") as fp:
            for i, line in enumerate(fp):
                if i == 2:
                    flag = 1
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
            leftxx=[]
            rightxx=[]
            leftyy=[]
            rightyy=[]
            #print('leftx len is '+ str(len(leftx)))
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
    
def calError(mypoints, file):
    Bleft, Bright, flag = readBenchMark(file)
    if flag == 1:
        BleftX = np.asarray(Bleft[0],dtype=np.float32)
        BleftY = np.asarray(Bleft[1])
        BrightX = np.asarray(Bright[0])
        BrightY = np.asarray(Bright[1])
        mypointsleftX = np.asarray(mypoints[0])
        mypointsrightX = np.asarray(mypoints[1])
        mypointsleftY = np.array(mypoints[2])
        mypointsrightY = np.array(mypoints[3])
        leftError = np.mean(np.sqrt((mypointsleftX-BleftX)**2 + (mypointsleftY-BleftY)**2))
        rightError = np.mean(np.sqrt((mypointsrightX-BrightX)**2 + (mypointsrightY-BrightY)**2))
    else:
        leftError = 0
        rightError = 0
    return leftError, rightError

framecounter = 0

direc = "./CU/CU1"
#direc = "./CU/CU2"
#direc = "./CU/CU3"
#direc = "./CU/CU4"
file_list = sorted(glob.glob(os.path.join(direc,"*.txt")))

########## INPUT VIDEO ##########
cap = cv.VideoCapture("./CU/CU1/CU1_5fps.mp4")
#cap = cv.VideoCapture("./CU/CU2/CU2_5fps.mp4")
#cap = cv.VideoCapture("./CU/CU3/CU3_5fps.mp4")
#cap = cv.VideoCapture("./CU/CU4/CU4_5fps.mp4")

########## OUTPUT VIDEO ##########
out = cv.VideoWriter('outputCU1.mp4',cv.VideoWriter_fourcc('m','p','4','v'), 5, (1640,590))
#out = cv.VideoWriter('outputCU2.mp4',cv.VideoWriter_fourcc('m','p','4','v'), 5, (1640,590))
#out = cv.VideoWriter('outputCU3.mp4',cv.VideoWriter_fourcc('m','p','4','v'), 5, (1640,590))
#out = cv.VideoWriter('outputCU4.mp4',cv.VideoWriter_fourcc('m','p','4','v'), 5, (1640,590))

# print(cap.isOpened())
while (cap.isOpened()):
    ret, frame = cap.read()
    
    # Performing Canny Operator
    grayscale = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    canny = cv.Canny(grayscale, 50, 150)
    #cv.imshow("canny", canny)

    # Performing Segmentation
    polygons = np.array([[(ROILEFT, ROIBOTTOM), (ROIRIGHT, ROIBOTTOM), (ROIMID, ROITOP)]])
    ROI = np.zeros_like(canny)
    cv.fillPoly(ROI, polygons, 255)
    segment = cv.bitwise_and(canny, ROI)
    cv.imshow("segment", segment)

    # Hough Transform
    hough = cv.HoughLinesP(segment, rho = 2, theta = np.pi / 180, threshold = 100, minLineLength = 100, maxLineGap = 40)
    #print(hough)

    # Calculating Lines
    lines = left_right_lines(frame, hough)
    #print(lines)

    # Overlaying Lines
    hough_overlay = np.zeros_like(frame)
    if lines is not None:
        for xstart, ystart, xend, yend in lines:
            #Thresholding maximum values for cv.line()
            if xstart > 32768:
                xstart = 32768
            if xend > 32768:
                xend = 32768
            if xstart < -32768:
                xstart = -32768
            if xend < -32768:
                xend = -32768
            cv.line(hough_overlay, (int(xstart), int(ystart)), (int(xend), int(yend)), (0, 255, 0), 5)
    
    #cv.imshow("hough", hough_overlay)
    
    output = cv.addWeighted(frame, 0.9, hough_overlay, 1, 1)
    cv.imshow("output", output)
    out.write(output)

    # Create 10 points along the line, including endpoints
    
    leftxstep = int(np.floor((lines[0][2]-lines[0][0])/10))
    leftystep = int(np.floor((lines[0][3]-lines[0][1])/10))
    rightxstep = int(np.floor((lines[1][2]-lines[1][0])/10))
    rightystep = int(np.floor((lines[1][3]-lines[1][1])/10))

    leftx = []
    lefty = []
    rightx = []
    righty = []

    for i in range(0,10):
        leftx.append(lines[0][0]+i*leftxstep)
        lefty.append(lines[0][1]+i*leftystep)
        rightx.append(lines[1][0]+i*rightxstep)
        righty.append(lines[1][1]+i*rightystep)
    
    #print("leftx: ")
    #print(leftx)

    mypoints = [leftx, rightx, lefty, righty]
    #print("mypoints:")
    #print(mypoints)
    
    # Calculating RMS error
    file = file_list[framecounter]
    leftError, rightError = calError(mypoints, file)
    #print("leftError: ", str(leftError))
    #print("rightError: ", str(rightError))

    #print(len(file_list))
    #print(framecounter)
    
    # Outputting error to text file
    with open("Naive_ErrorOP_CU1.txt","a") as text_file:
        text_file.write("%f %f " % (leftError, rightError))

    if cv.waitKey(10) & 0xFF == ord('q'):
        break

    framecounter = framecounter+1

cap.release()
cv.destroyAllWindows()

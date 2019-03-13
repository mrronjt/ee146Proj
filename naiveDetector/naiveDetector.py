import cv2 as cv
import numpy as np

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

framecounter = 0

cap = cv.VideoCapture("crowded_lane_switching.MP4")
out = cv.VideoWriter('output_crowded_lane_switching.mp4',cv.VideoWriter_fourcc('m','p','4','v'), 5, (1640,590))
while (cap.isOpened()):
    ret, frame = cap.read()
    
    # Performing Canny Operator
    grayscale = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    canny = cv.Canny(grayscale, 50, 150)
    cv.imshow("canny", canny)

    # Performing segmentation
    polygons = np.array([[(ROILEFT, ROIBOTTOM), (ROIRIGHT, ROIBOTTOM), (ROIMID, ROITOP)]])
    ROI = np.zeros_like(canny)
    cv.fillPoly(ROI, polygons, 255)
    segment = cv.bitwise_and(canny, ROI)
    cv.imshow("segment", segment)

    # Hough transform
    #hough = cv.HoughLinesP(segment, 2, np.pi / 180, 100, np.array([]), minLineLength = 100, maxLineGap = 40)
    hough = cv.HoughLinesP(segment, rho = 2, theta = np.pi / 180, threshold = 100, minLineLength = 100, maxLineGap = 40)
    #print(hough)
    lines = left_right_lines(frame, hough)
    #print(lines)

    #hough_overlay = overlay(frame, lines)
    hough_overlay = np.zeros_like(frame)
    if lines is not None:
        for xstart, ystart, xend, yend in lines:
            if xstart > 32768:
                xstart = 32768
            if xend > 32768:
                xend = 32768
            if xstart < -32768:
                xstart = -32768
            if xend < -32768:
                xend = -32768
            print(xstart, ystart, xend, yend)
            cv.line(hough_overlay, (int(xstart), int(ystart)), (int(xend), int(yend)), (0, 255, 0), 5)
    
    #cv.imshow("hough", hough_overlay)
    
    output = cv.addWeighted(frame, 0.9, hough_overlay, 1, 1)
    cv.imshow("output", output)
    out.write(output)

    # TODO: Add error evaluation code
    print(lines)

    if cv.waitKey(10) & 0xFF == ord('q'):
        break

    framecounter = framecounter+1

cap.release()
cv.destroyAllWindows()

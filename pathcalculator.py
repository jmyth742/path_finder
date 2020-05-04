#ROBOCUP 19/20 final assignment - 4 people in team
# 2 people working on open cv and images
#1 person on the kinematics of the robot
#1 person working on the code of the tic tac toe game logic.
# 
# Jonathan Smyth
# EIT Digital double degree program, exit year.
# TUBerlin
# WS19/20
# student number - 0415261 

## NOTES
# This file is used to work out the distance and plot a path for the robot
# it is assumed that the mid point of the bottom of the image is where the robot
# is standing at all point, the x component and the y component of the path is plotted
# this is then returned to the robot to actually move in those direction.
# the values of the centroid of the tic tac toe grid are in this case 
# hard coded so the job can be done.
# to reference the distance an object of known size is places in the centre bottom
# where the robot would normally have its 'feet'
# 

# import the necessary packages
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
from math import sqrt
import numpy as np
import argparse
import imutils
import cv2

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def calculatePath(goal_pos):
	finish_x, finish_y = centres[goal_pos][:]
	print ("finsih coords", finish_x, finish_y)
	
	if finish_x > feet_x:
		x_dist = finish_x - feet_x
	else:
		x_dist = feet_x - finish_x
	
	if finish_y > feet_y:
		y_dist = finish_y - feet_y
	else:
		y_dist = feet_y - finish_y
	print(x_dist)	
	print(y_dist)

	cv2.line(orig, (int(feet_x),int(feet_y)), (int(finish_x),int(feet_y)), (9,9,9), 3)
	cv2.line(orig, (int(finish_x),int(feet_y)), (int(finish_x),int(finish_y)), (9,9,9), 3)
	cv2.imshow("Image", orig)
	cv2.waitKey(0)

	euc_x = x_dist * scale	
	euc_y = y_dist * scale

	return euc_x, euc_y



# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
ap.add_argument("-w", "--width", type=float, required=True,
	help="width of the left-most object in the image (in m)")
ap.add_argument("-p", "--position", type=float, required=True,
	help="The position in the tic tac toe grid, 0-8 tl->bl->mt->mb->tr->br")
args = vars(ap.parse_args())

# load the image, convert it to grayscale, and blur it slightly
image = cv2.imread(args["image"])
cv2.imshow("orig", image)
cv2.waitKey(0)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("greyed img", gray)
cv2.waitKey(0)
gray = cv2.GaussianBlur(gray, (7, 7), 0)
cv2.imshow("gausian blur", gray)
cv2.waitKey(0)

# perform edge detection, then perform a dilation + erosion to
# close gaps in between object edges
edged = cv2.Canny(gray, 50, 100)
edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)
cv2.imshow("edged", edged)
cv2.waitKey(0)

dims = image.shape

feet_y = image.shape[0]
feet_x = image.shape[1]/2

print(feet_x, feet_y)


centres = np.zeros((9,2))
mids = np.zeros((2,1))


# find contours in the edge map
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# sort the contours from left-to-right and, then initialize the
# distance colors and reference object
(cnts, _) = contours.sort_contours(cnts)
colors = ((0, 0, 255), (240, 0, 159), (0, 165, 255), (255, 255, 0),
	(255, 0, 255))
refObj = None




# loop over the contours individually
# to get the centre points of each of our grids
# we then use these to work out which positon is which 
count = 0
for c in cnts:

	#print(cv2.contourArea(c))
	# if the contour is not sufficiently large, ignore it
	if cv2.contourArea(c) > 15000.0:
		continue

	# compute the rotated bounding box of the contour
	box = cv2.minAreaRect(c)
	box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
	box = np.array(box, dtype="int")

	# order the points in the contour such that they appear
	# in top-left, top-right, bottom-right, and bottom-left
	# order, then draw the outline of the rotated bounding
	# box
	box = perspective.order_points(box)

	# compute the center of the bounding box
	#print("box ", box[:,0], box[:,1])
	cX = np.average(box[:, 0])
	cY = np.average(box[:, 1])


	# draw the contours on the image
	orig = image.copy()
	cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
	#cv2.drawContours(orig, [refObj[0].astype("int")], -1, (0, 255, 0), 2)

	
	objCoords = np.vstack([box, (cX, cY)])
	midx,midy = objCoords[4,:]
	mids = midx,midy

	
	##some formatting here, to get the ref obj distance etc. 
	if count == 4:
		x = 300.0
		y = 212.5
		points = x,y
		centres[count,:] = points
		centres[count+1,:] = mids
	elif count == 0 :
		#we use the first reference box as the our scale as we know what size it is
		# then we can scale the rest of the pixels to calculate the distance in m
		centres[count,:] = mids
		(tl, tr, br, bl) = box
		marker = tr - tl
		# print (tl, tr, br, bl)
		# print (marker)
		# print box
		(tlblX, tlblY) = midpoint(tl, bl)
		(trbrX, trbrY) = midpoint(tr, br)
		D = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
		scale = args["width"] / D
		# refObj = (box, (feet_x, feet_y), D / args["width"])
		# print(refObj[2])
	elif count > 4:
		centres[count+1,:] = mids
	else:
		centres[count,:] = mids

	color = (0, 0, 255)

	cv2.circle(orig, (int(feet_x), int(feet_y)), 5, color, -1)
	if count == 4:
		cv2.circle(orig, (int(centres[4][0]), int(centres[4][1])), 5, color, -1)

	cv2.circle(orig, (int(midx), int(midy)), 5, color, -1)

	cv2.imshow("Image", orig)
	cv2.waitKey(0)
	count+=1


print("Here is our list of centroid points")
print(centres)

print("Distance to be travelled in X component    ", calculatePath(7), " and finally then Y component")




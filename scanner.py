# Document scanner

import numpy as np
import cv2
from datetime import datetime

path = "Images/document.jpg"    # path of the document image

widthImg = 480
heightImg = 640

# finds the edges of the image
def getContours(img):
    biggest = np.array([])
    maxArea = 0
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area>5000:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
            if area>maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area
    return biggest

# the image is preprocessed by applying different filters, to find the edges and the text
def preProcessing(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # turning into gray scale image
    imgBlur = cv2.GaussianBlur(imgGray, (5,5), 1) # adding gausian blur
    thres = valTrackbars() # getting track bar values
    imgCanny = cv2.Canny(imgBlur, thres[0], thres[1]) # canny blur
    kernel = np.ones((5,5))
    imgDial = cv2.dilate(imgCanny, kernel, iterations=1) # applying dilation
    imgErode = cv2.erode(imgDial, kernel, iterations=1) # applying erosion
    return imgErode

# calculates the four courner points of the image
def reorder(myPoints):
    myPoints = myPoints.reshape((4,2))
    myPointsNew = np.zeros((4, 1, 2), np.int32)
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew

# the image is cropped at the edges, with the obtained four points(co-ordinates)
def getWarp(img, biggest):
    biggest = reorder(biggest)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0,0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgOutput = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
    imgOutput = imgOutput[5:img.shape[0]-5, 5:img.shape[1]-5]
    imgOutput = cv2.resize(imgOutput, (480, 640))
    return imgOutput

def nothing(x):
    pass
 
# this will open a trackbar window with which we can adjust our document scan
def initializeTrackbars():
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 360, 120)
    cv2.createTrackbar("Threshold1", "Trackbars", 10,255, nothing)
    cv2.createTrackbar("Threshold2", "Trackbars", 50, 255, nothing)
    
# adjust the threshold values to get the desired scan
def valTrackbars():
    Threshold1 = cv2.getTrackbarPos("Threshold1", "Trackbars")
    Threshold2 = cv2.getTrackbarPos("Threshold2", "Trackbars")
    src = Threshold1,Threshold2
    return src
  
initializeTrackbars()

while True:
    img = cv2.imread(path)
    img = cv2.resize(img, (widthImg, heightImg)) # resizing the image
    cv2.imshow("Original", img)
    imgProcessed = preProcessing(img)
    biggest = getContours(imgProcessed)

    if biggest.size != 0:
        imgWarp = getWarp(img, biggest)
        
        # applying adaptive threshold
        imgWarpGray = cv2.cvtColor(imgWarp,cv2.COLOR_BGR2GRAY)
        imgAdaptiveThre= cv2.adaptiveThreshold(imgWarpGray, 255, 1, 1, 7, 2)
        imgResult = cv2.bitwise_not(imgAdaptiveThre)

    else:
        # blank image
        imgResult = np.zeros((heightImg,widthImg, 3), np.uint8)
        
    # we get the three forms of the scanned document
    cv2.imshow('warp', imgWarp)  # coloured document
    cv2.imshow('gray', imgWarpGray) # gray scaled document
    cv2.imshow('scanned', imgResult) # highlighted document
    
    # press 's' to save the scanned document in the "Scanned" folder
    if cv2.waitKey(1) & 0xFF == ord('s'):
        name = path.split('/')[-1].split('.')[0]
        time = datetime.now().strftime('%H-%M-%S')
        
        # saving the all three formats
        cv2.imwrite("Scanned/"+name+"_warp"+str(time)+".jpg", imgWarp)
        cv2.imwrite("Scanned/"+name+"_gray"+str(time)+".jpg", imgWarpGray)
        cv2.imwrite("Scanned/"+name+"_scanned"+str(time)+".jpg", imgResult)
        
        # Prints "Scan saved" on the original image window to ensure that your scan saved
        cv2.rectangle(img, (0, 200), (480, 300), (255,0,0),cv2.FILLED)
        cv2.putText(img, "Scan Saved", (75, 265), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 0), 2)
        cv2.imshow("Original", img)
        cv2.waitKey(500)
        break

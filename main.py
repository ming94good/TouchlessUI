import cv2
import numpy as np
import copy
import math
from mss import mss
import time
import vlc

# parameters
cap_region_x_begin=0.5  # start point/total width
cap_region_y_end=0.8  # start point/total width
threshold = 60  #  BINARY threshold
blurValue = 41  # GaussianBlur parameter
bgSubThreshold = 50
learningRate = 0

# variables
isBgCaptured = 0   # bool, whether the background captured

# songs
song1 = 'song1.mp3'
song2 = 'song2.mp3'
song3 = 'song3.mp3'

def printThreshold(thr):
    print("Changed threshold to "+str(thr))


def removeBG(frame):
    fgmask = bgModel.apply(frame,learningRate=learningRate)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # res = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res


# Camera
camera = cv2.VideoCapture(0)
camera.set(10,200)
cv2.namedWindow('trackbar')
cv2.createTrackbar('trh1', 'trackbar', threshold, 100, printThreshold)

## setup vlc player list
player = vlc.Instance()
media_list = player.media_list_new()
media_player = player.media_list_player_new()
media1 = player.media_new("song1.mp3")
media2 = player.media_new("song2.mp3")
media3 = player.media_new("song3.mp3")
media_list.add_media(media1)
media_list.add_media(media2)
media_list.add_media(media3)
media_player.set_media_list(media_list)

while camera.isOpened():
    ret, frame = camera.read()
    threshold = cv2.getTrackbarPos('trh1', 'trackbar')
    frame = cv2.bilateralFilter(frame, 5, 50, 100)  # smoothing filter
    frame = cv2.flip(frame, 1)  # flip the frame horizontally
    cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
                 (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)
    height, width, channels = frame.shape
    cv2.imshow('original', frame)

    #  Main operation
    if isBgCaptured == 1:  # this part wont run until background captured
        img = removeBG(frame)
        img = img[0:int(cap_region_y_end * frame.shape[0]),
                    int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]  # clip the ROI
        #cv2.imshow('mask', img)

        # convert the image into binary image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
        #cv2.imshow('blur', blur)
        ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)
        #cv2.imshow('ori', thresh)


        # get the coutours      
        thresh1 = copy.deepcopy(thresh)
        contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        length = len(contours)
        maxArea = -1
        if length > 0:
            #get highest coordinate from countour
            c = max(contours, key=cv2.contourArea)
            extTop = tuple(c[c[:, :, 1].argmin()][0])
            #print(extTop) 
            
            for i in range(length):  # find the biggest contour (according to area)
                temp = contours[i]
                area = cv2.contourArea(temp)
                if area > maxArea:
                    maxArea = area
                    ci = i
                    
            res = contours[ci]
            hull = cv2.convexHull(res)
            drawing = np.zeros(img.shape, np.uint8)
            cv2.drawContours(drawing, [res], 0, (0, 255, 0), 2)
            cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3)
            
            # setup menu control
            if (abs(extTop[0]-(40,40)[0]) < 5 ) & (abs(extTop[1]-(40,40)[1]) < 5):
                media_player.play() # play music from playlist
                print('play')
                time.sleep(1)
            elif (abs(extTop[0]-(150,40)[0]) < 5 ) & (abs(extTop[1]-(150,40)[1]) < 5):
                media_player.stop() # stop playing music
                print('stop')
                time.sleep(1)
            elif (abs(extTop[0]-(270,40)[0]) < 5 ) & (abs(extTop[1]-(270,40)[1]) < 5):
                media_player.next() # play next song from playlist
                print('next')
                time.sleep(1)           
        
        # draw menu interface
        cv2.imshow('output', drawing)
        circle1 = cv2.circle(drawing, (40, 40), 20, (255, 0, 0), 2)
        circle2 = cv2.circle(drawing, (150, 40), 20, (255, 0, 0), 2)
        circle3 = cv2.circle(drawing, (270, 40), 20, (255, 0, 0), 2)
        cv2.imshow('output', circle1)
        cv2.imshow('output', circle2)
        cv2.imshow('output', circle3)
        text1= cv2.putText(drawing, 'Play', (20, 46), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)
        text2= cv2.putText(drawing, 'Stop', (120, 46), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)
        text3= cv2.putText(drawing, 'Next', (235, 46), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow('output', text1)
        cv2.imshow('output', text2)
        cv2.imshow('output', text3)
        
    # Keyboard OP
    k = cv2.waitKey(10)
    if k == 27:  # press ESC to exit
        camera.release()
        cv2.destroyAllWindows()
        break
    elif k == ord('b'):  # press 'b' to capture the background
        bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
        isBgCaptured = 1
        print('Background Captured')
    elif k == ord('r'):  # press 'r' to reset the background
        bgModel = None
        isBgCaptured = 0
        print ('Reset BackGround')
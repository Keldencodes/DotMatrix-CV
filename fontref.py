# For locating and identifying dot matrix digits
# Kelden Ben-Ora Dec 2019

from PIL import Image
from PIL.ExifTags import TAGS
from imutils import contours
from imutils import paths
from imutils.perspective import four_point_transform
import numpy as np
import argparse
import imutils
import cv2
import os

dir = './dataPics'

# retrieve the date and time at which the pic was taken
def get_exif(fn):
    ret = ''
    i = Image.open(fn)
    info = i._getexif()
    for tag, value in info.items():
        decoded = TAGS.get(tag, tag)
        if decoded == 'DateTimeDigitized':
            ret = value
        
    return ret


# process reference image then isolate and identify digits

ref = cv2.imread('ref.jpg')
ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
ref = cv2.threshold(ref, 10, 255, cv2.THRESH_BINARY)[1]
# cv2.imshow('threshed',ref)
# cv2.waitKey(0)
refCnts, hierarchy = cv2.findContours(ref, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
test = cv2.imread('ref.jpg')
# test = cv2.drawContours(cv2.imread('ref.jpg'), refCnts, -1, (0,255,0), 3)
# cv2.imshow('contours',test)
# cv2.waitKey(0)
refCnts = contours.sort_contours(refCnts, method="left-to-right")[0]
# print 'num of contours: %r' % len(refCnts)
# for i in range(len(refCnts)):
#     for j in range(len(refCnts[i])):
#         print "(i:%r, j:%r)" % (i,j)
#         print refCnts[i][j][0]
    
digits = {}

for (i,c) in enumerate(refCnts):
    (x,y,w,h) = cv2.boundingRect(c)
    test = cv2.rectangle(test,(x,y),(x+w,y+h),(0,255,0),3)
    roi = ref[y:y+h, x:x+w]
    roi = cv2.resize(roi, (57,88))
    digits[i] = roi
cv2.imshow('reference',test)
cv2.waitKey(0)

################################
# process main images in a dir #
################################

for root, dirs, files in os.walk(dir, topdown=False):
    for name in files:
        dir = dir[2:len(dir)]
        fn = dir+'/'+name
        print fn
        image = cv2.imread(fn)
        image = imutils.resize(image, height=900)
        cv2.imshow('image',image)
        cv2.waitKey(0)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        edged = cv2.Canny(gray.copy(), 50, 200, 255)
        
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        
        # found rectangular lcd 
        
        displayCnt = None

        for c in cnts:
            peri = cv2.arcLength(c,True)
            approx = cv2.approxPolyDP(c, 0.02*peri, True)
            if len(approx) == 4:
                displayCnt = approx
                break
                
        gray = four_point_transform(gray, displayCnt.reshape(4,2))
        warped = four_point_transform(image, displayCnt.reshape(4,2))

        rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))

        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 1)

        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = contours.sort_contours(cnts, method="left-to-right")[0]

        output = []
        
        for (i,c) in enumerate(cnts):
    
            (x,y,w,h) = cv2.boundingRect(c)
    
            if (w > 5) and (h > 5) and h > w:
        
                test2 = cv2.rectangle(test0,(x,y),(x+w,y+h),(0,255,0),1)
                roi = thresh[y:y + h, x:x + w]
                roi = cv2.resize(roi, (57,88))
                scores = []
        
                for (digit, digitROI) in digits.items():
            
                    result = cv2.matchTemplate(roi, digitROI, cv2.TM_CCOEFF)
                    (_,score,_,_) = cv2.minMaxLoc(result)
                    scores.append(score)
            
                output.append(str(np.argmax(scores)))
                test2 = cv2.putText(test2, str(np.argmax(scores)), (x, y+4), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,255), 2)
        
        output = int(''.join(output))
        print 'output: %r' % output
        print 'Date and time: %r ' % get_exif(fn)

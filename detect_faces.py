#Importing the necessary libraries
import numpy as np
import cv2
import os
import sys
import time
#creating a face cascade
FacesCascade = cv2.CascadeClassifier('C:/Users/Home/Desktop/project/haarcascade_frontalface_default.xml')
#setting video source to default
webcam = cv2.VideoCapture(0)
#webcam.set(3,640)
#webcam.set(4,480) 
while True:
    #caoturing image frame by frame
    retur, image = webcam.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = FacesCascade.detectMultiScale(
        gray,     
        scaleFactor=1.2,
        minNeighbors=5,     
        minSize=(30, 30)
    )
    #returns position of the face as a rectangle
    for (a,b,w,h) in faces:
        cv2.rectangle(image,(a,b),(a+w,b+h),(255,0,0),2) 
    cv2.imshow('video',image)
    
    k = cv2.waitKey(20) #& 0xff
    if k == 27: 
        break
webcam.release()
cv2.destroyAllWindows()
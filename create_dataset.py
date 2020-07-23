
import cv2
import numpy
import sys
import os
import time
#loading the xml files to detect faces
detector = 'C:/Users/Home/Desktop/project/haarcascade_frontalface_default.xml'
#path to the folder where the images will be stored 
datasets = 'C:/Users/Home/Desktop/project/dataset'
#a unique subset will be created for every person, 
#the name of this sub dataset is changed for every person
subset = 'Aditee'

#concatenating the paths to store subset as a folder in datasets
path = os.path.join(datasets, subset)
#if the subset folder is not created already, then a subset is created by the name initialised above
if not os.path.isdir(path):
    os.mkdir(path)


(width, height) = (130, 100)
#size of the image being stored
FacesCascade = cv2.CascadeClassifier(detector)
#calling the cdefault camera
webcam = cv2.VideoCapture(0)

c = 1
print("Taking pictures, change face angles")
#The loop will excecute 50 times and take 50 pictures of the person infront of camera
while c < 51:
    #image=camera stream
    retur, image = webcam.read()
    # if webcam is accessed
    if retur == True:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # face detection is performed using haar cascade files
        #passing some important parameters
        faces = FacesCascade.detectMultiScale(gray, 1.3, 4)
        for (x,y,w,h) in faces:
            # draws a rectangle around your face when taking pictures
            # this is done so that it only takes pictures of your face
            cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
            face = gray[y:y + h, x:x + w]
            # resize the face images to the size of the 'face' variable above
            face_resize = cv2.resize(face, (width, height))
            # save image in the folder with it's corresponding number
            cv2.imwrite('%s/%s.png' % (path,c), face_resize)
        c += 1
        # display the openCV window
        cv2.imshow('OpenCV', image)
        key = cv2.waitKey(20)
         #press esc to stop the loop
        if key == 27:
            break
print("Subset created.")
webcam.release()
cv2.destroyAllWindows()
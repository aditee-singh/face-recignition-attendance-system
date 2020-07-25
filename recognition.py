#making necessary imports
size = 4
import cv2
import sys
import numpy
import os

#loading the haar file, change this path to the path where the file is stored
haar_file = 'C:/Users/Home/Desktop/project/haarcascade_frontalface_default.xml'
#path to the dataset
datasets = 'C:/Users/Home/Desktop/project/dataset'

print('Training classifier, this may take a few seconds')

(images, labels, names, id) = ([], [], {}, 0)
#creating a list of images and a list of their names along with a unique id
for (subdirs, dirs, files) in os.walk(datasets):
    for subdir in dirs:
      names[id] = subdir
    #person's name used as subset using creat_dataset.py
      subjectpath = os.path.join(datasets, subdir)
      for filename in os.listdir(subjectpath):
            path = subjectpath + '/' + filename
            label = id
            images.append(cv2.imread(path, 0))
            labels.append(int(label))
      id += 1
(width, height) = (130, 100)
#function to draw a fancy rounded rectangle around the face
#def rounded_rectangle(img, pt1, pt2, color, thickness, r, d):
 #   x1,y1 = pt1
 #   x2,y2 = pt2
 
    # Top left
  #  cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
   # cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    #cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
 
    # Top right
    #cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    #cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    #cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
 
    # Bottom left
    #cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    #cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    #cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
 
    # Bottom right
    #cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    #cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    #cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)
#for (a,b,w,h) in images:
 #   cv2.rectangle(img,(a,b),(a+w,b+h),(255,0,0),2)
# making a numpy array from the above lists
(images, labels) = [numpy.array(lists) for lists in [images, labels]]

#training a model from the images using Local Binary Patterns algorithm on the images and labels above
model = cv2.face.LBPHFaceRecognizer_create()

model.train(images, labels)

face_cascade = cv2.CascadeClassifier(haar_file)
webcam = cv2.VideoCapture(0)
print('Classifier trained, now recognising faces.')

while True:
    (_, im) = webcam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # detect faces using the haar_cacade file
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
    
    # draw a rectangle around the face and resizing/ grayscaling it
    
        cv2.rectangle(im,(x,y),(x + w,y + h),(0, 255, 255),2)
        face = gray[y:y + h, x:x + w]
        sample = cv2.resize(face, (width, height))
        # try to recognize the face(s) using the resized faces we made above
        recognized = model.predict(sample)
        #rounded_rectangle(im, (x, y), (x + w, y + h), (0, 255, 255), 2, 15, 30)
        # when face is recognized 
        if recognized[1] < 74:
            cv2.putText(im,'%s' % (names[recognized[0]].strip()),(x + 5, (y + 25) + h), cv2.FONT_HERSHEY_PLAIN,1.5,(20,185,20), 2)
        #find accuracy percentage
            accuracy = (recognized[1]) if recognized[1] <= 100.0 else 100.0

            #print person's name and accuracy percentage in standard output
            #print("person: {}, accuracy: {}%".format(names[recognized[0]].strip(), round((accuracy / 74.5) * 100, 2)))
        # if face is not found in the dataset, print unknown 
        else:
            cv2.putText(im,'Unknown',(x + 5, (y + 25) + h), cv2.FONT_HERSHEY_PLAIN,1.5,(65,65, 255), 2)
            #print("predicted person: Unknown")

    # show window and set the window title
    cv2.imshow('OpenCV Face Recognition -  esc to close', im)
    key = cv2.waitKey(10)
    # esc to quit applet
    if key == 27:
        break

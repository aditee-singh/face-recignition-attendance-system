#Importing the necessary libraries
import face_recognition
import cv2
import numpy as np
import os
import click
import re
import multiprocessing
import glob

#Accesing the default webcam
video_capture = cv2.VideoCapture(0)

#initialising variables to store details of saved images
imagesencodings = []
imagesnames = []
dirname = os.path.dirname(__file__)
path = os.path.join(dirname, 'images/')

#accesing the saved images
savedimages = [f for f in glob.glob(path+'*.jpg')]

#storing the number of saved images
numberofsavedimages = len(savedimages)

#creating a duplicate list of the saved images
names = savedimages.copy()

#accesing the saved images and extracting their encodings by invoking the face_recognition library
for i in range(numberofsavedimages):
    globals()['image_{}'.format(i)] = face_recognition.load_image_file(savedimages[i])
    globals()['image_encoding_{}'.format(i)] = face_recognition.face_encodings(globals()['image_{}'.format(i)])[0]

    imagesencodings.append(globals()['image_encoding_{}'.format(i)])


    names[i] = names[i].replace("images/", "")  
    imagesnames.append(names[i])

unkownlocations = []
unknownencodings = []
face_names = []
process = True
#These varoables have been declared to process the unknown face

while True:
    
    ret, frame = video_capture.read()
    #capturing one frame from the video
    
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    #captured frame is resized to 1/4th make thr process faster

    rgb_small_frame = small_frame[:, :, ::-1]
    #the frame is coverted from BGR color used by oprnCV to RGB color used by face_recognition

    
    if process:
        unknownlocations = face_recognition.face_locations(rgb_small_frame)
        unknownencodings = face_recognition.face_encodings(rgb_small_frame, unknownlocations)
        #Identifying the faces and their encodings in the fram captured

        face_names = []
        for face_encoding in unknownencodings:
        
            matches = face_recognition.compare_faces(imagesencodings, face_encoding)
            name = "Unknown"
            #checking for the match of the identified face in savedimages
    
            face_distances = face_recognition.face_distance(imagesencodings, face_encoding)
            matchingface = np.argmin(face_distances)
            if matches[matchingface]:
                name = imagesnames[matchingface]

            face_names.append(name)
             
             #If a match is found in the saved images, using the face with the smallest distace to the new face
             
    process = not process

          
    for (top, right, bottom, left), name in zip(unknownlocations, face_names):
        #Displaying the results

        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        #Rescaling the image to size 4 times because we reduced it to 1/4th earlier

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        #Drawing a label with a name below the face

    cv2.imshow('Video', frame)
    #displaying the resulting image

    if cv2.waitKey(1) & 0xFF == ord('a'):
        break
    #Program stops when 'a' is pressedn on the keyboard

video_capture.release()
cv2.destroyAllWindows()
#releasing the webcam
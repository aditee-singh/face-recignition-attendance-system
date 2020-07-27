# face-recognition-attendance-system
Face recognition system built on opencv 4.3 and Python 3.8 code to detect faces using Haar Cascade and match facsed using LBPH (Local Binary Patterns Histogram) Recognition on live webcam.
#Prerequisites:
1. Python 3.8
2. Opencv 4.3 
3. Anaconda 2020.7
Python libraries - numpy, sckilit, dlib.
#How to Use: (Step by Step)
1. Save the "haarcascade_frontalface_default.xml" file, create_dataset.py, recognition.py files in your directory(say project).
2. Create a sub-directory with the name "dataset".
3. Run the create_dataset.py file to create your own dataset, Change the name of the subset in the code to person's name.
This file uses the Haar Cascade Classifiers to detect faces in the frame. a sub-directory is created under the "datasets" folder by the new label (the name of your subset). To add pictures of the same person, use the same label name when prompted. Simply change the label name to save pictures of a new face. Another directory in 'datasets' is created. Hence each person gets a directory in 'dataset' to store our images in an ordered and organised manner.
4. Run the recognition.py file to recognise faces using live webcam. Detecting the faces Once some data is added in the dataset directory, the program automatically trains using the LBPH Face Recognition. The recognizer is trained on the selected cropped faces from the dataset. The faces in front of the web cam are used to predict labels on the trained model. Set a minimum confidence score to display labels under detected faces. The faces in front of the web cam are automatically labelled over the video window according to the predictions. A confidence score is also displayed over the frame.
Press esc key to exit the program.


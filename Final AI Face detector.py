import cv2  #  open source computer vision library
import random # imports the random functions for random integers and what no, when in use use random.(fill in teh function)

trained_face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # setting up variable for face trained data. cascade classifier is the cv2 function for recognizing something
trained_left_eye = cv2.CascadeClassifier('haarcascade_lefteye_2splits.xml')
trained_right_eye = cv2.CascadeClassifier('haarcascade_righteye_2splits.xml')
# img = cv2.imread('Pic.jpg') #  opencv function for reading an image
# read in a binary array from pixels to numbers

defaultwebcam = cv2.VideoCapture(0) # Created a variable reading webcam input

# Iterate over frames forever
while True:

    # Reads the current frame
    successful_frame_read, frame = defaultwebcam.read() # the read function is returning a tuple of values, one variable is a boolean telling you if it is properly reading the frames, and the other

    # Grayscale image
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # creates variable representation of the original image but making it grayscale

    # Detect Faces
    face_coordinates = trained_face.detectMultiScale(grayscaled_img)
    face_coordinates_Reye = trained_right_eye.detectMultiScale(grayscaled_img)
    face_coordinates_Leye = trained_left_eye.detectMultiScale(grayscaled_img)
    # trained_face.detectMultiScale(grayscaled_img)
    # uses the xml file trained data to detect faces, next step will use this variable to place the box around the face
    # detect multi scale is the function that uses the orignal variable to detect faces at any size.
    # gives the coordinates of the rectangles around the face

    for (x, y, w, h) in face_coordinates:
        # sets up a for loop for the face box maker, and it goes through the arrays and places squares around them
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
    for (x,y,w,h) in face_coordinates_Reye:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow('Facial Detector', frame)
    key = cv2.waitKey(1)


    if key == 81:  # Press q + shift to close application
        break

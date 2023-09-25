# python --version
# XX python version 3.9
# pip --version
# XX pip version 21.0.1
# pip install --upgrade pip
# pip install face-recognition
# pip install opencv-python

## THIS DOES NOT TRAIN -- TRAIN THE PICfirst and save as train.pkl
## use the file windowsTrainSaveKnownFaces

## FPS rate comes out as 0.1 running on my laptop


import face_recognition
import cv2
import os
import pickle
import time
print(cv2.__version__)

Encodings=[]
Names=[]

with open('train.pkl','rb') as f:
    Names=pickle.load(f)
    Encodings=pickle.load(f)
font=cv2.FONT_HERSHEY_SIMPLEX


cam= cv2.VideoCapture(0) # 0 for built in camera, if USB change it to 1

#FPS COUNT
timeMark=time.time()
fpsFilter=0

while True:

    _,frame=cam.read()
    frameSmall=cv2.resize(frame,(0,0),fx=.25,fy=.25)
    frameRGB=cv2.cvtColor(frameSmall,cv2.COLOR_BGR2RGB)
    facePositions=face_recognition.face_locations(frameRGB,model='cnn')
    allEncodings=face_recognition.face_encodings(frameRGB,facePositions)
    for (top,right,bottom,left),face_encoding in zip(facePositions,allEncodings):
        name='Unkown Person'
        matches=face_recognition.compare_faces(Encodings,face_encoding)
        if True in matches:
            first_match_index=matches.index(True)
            name=Names[first_match_index]
        top=top*4
        right=right*4
        bottom=bottom*4
        left=left*4
        # FPS TIME CALCUALATION
        dt=time.time()-timeMark
        timeMark=time.time()
        fps=1/dt
        fpsFilter=.95*fpsFilter + .05 *fps
        cv2.rectangle(frame,(left,top),(right, bottom),(0,0,255),2)
        cv2.putText(frame,name + str(round(fpsFilter,1))+ 'fps',(left,top-6),font,.75,(0,0,255),2)
    cv2.imshow('Picture',frame)
    cv2.moveWindow('Picture',0,0)
    if cv2.waitKey(1)==ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
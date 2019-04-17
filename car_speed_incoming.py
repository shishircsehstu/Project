#import libraries of python opencv
import math
import cv2
import numpy as np

x1 = 250
x2 = 1500
y1 = 160
y2 = 180

x3 = 0
x4 = 1500
y3 = 360
y4 = 380



#create VideoCapture object and read from video file
cap = cv2.VideoCapture('car.mp4')
#use trained cars XML classifiers
car_cascade = cv2.CascadeClassifier('cars.xml')

fps = cap.get(cv2.CAP_PROP_FPS)
print(fps)

ar = []
fr_no = 0
min_d1 = 1000
min_d2 = 1000
line_2_car = 0

#read until video is completed
while True:
    fr_no = fr_no + 1
    #capture frame by frame
    ret, frame = cap.read()
    #convert video into gray scale of each frames
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #detect cars in the video
    cars = car_cascade.detectMultiScale(gray, 1.3, 3)

    #to draw arectangle in each cars


    for (x,y,w,h) in cars:
        cv2.line(frame,(x1,y1),(x2,y2),(0,255,255),2)
        cv2.line(frame,(x3,y3),(x4,y4),(0,255,255),2)
        
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

        px = x+w/2.0;
        py = y+h/2.0;

#------------- detect in first line

        a = (y1-y2)
        b = (x2-x1)
        c = x1*y2-x2*y1

        d = (a*px+b*py+c)/ math.sqrt(a*a+b*b)

        if py>y1:
            if d<min_d1:
                ar.append(fr_no)
            min_d1 = d

            
#------------- detect in second line
        a = (y3-y4)
        b = (x4-x3)
        c = x3*y4-x4*y3

        d = (a*px+b*py+c)/ math.sqrt(a*a+b*b)

        if py>y4:
            if d<min_d2:
                line_2_car += 1;
                frame_diff = fr_no - ar[line_2_car-1]
                t = frame_diff/fps;
                if t==0:
                    v = 0;
                else:
                    v = 10/t;

                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame,str(v),(200,200),font,1,(200,255,255),2,cv2.LINE_AA)
            min_d2 = d
    #display the resulting frame
    cv2.imshow('video', frame)
    #press Q on keyboard to exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
#release the videocapture object
cap.release()
#close all the frames
cv2.destroyAllWindows()

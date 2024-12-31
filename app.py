from flask import Flask,render_template,request,redirect,url_for,Response
import cv2
from flask.wrappers import Response
from keras.models import load_model
import os
import numpy as np
from pygame import mixer
import time

from tensorflow.python.platform.tf_logging import debug

app=Flask(__name__)
cam = cv2.VideoCapture(0)
mixer.init()
sound=mixer.Sound('alarm.wav')
sound.play()
time.sleep(1)
sound.stop()

face=cv2.CascadeClassifier(r'haar cascade files\haarcascade_frontalface_alt.xml')
leye=cv2.CascadeClassifier(r'haar cascade files\haarcascade_lefteye_2splits.xml')
reye=cv2.CascadeClassifier(r'haar cascade files\haarcascade_righteye_2splits.xml')

label=['closed','open','no_yawn','yawn']

model=load_model('driver drowsiness detection.h5')

path = os.getcwd()
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count=0
score=0
thicc=2
rpred=[99]
lpred=[99]
path = os.getcwd()


    
def get_frames(cam,leye,reye,count,score,thicc,font,face,path,rpred,lpred):
    while True:
        success,frame=cam.read()

        if not success:
            break
        else:
            #while True:
                
            #ret, frame = cam.read()
            height,width = frame.shape[:2] 

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
            left_eye = leye.detectMultiScale(gray)
            right_eye =  reye.detectMultiScale(gray)

            cv2.rectangle(frame, (0,height-50) , (200,height) , (225,225,0) , thickness=cv2.FILLED )

            for (x,y,w,h) in faces:
                cv2.rectangle(frame, (x,y) , (x+w,y+h) , (100,100,100) , 1 )

            for (x,y,w,h) in right_eye:
                r_eye =frame[y:y+h,x:x+w]
                count =count+1
                r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
                r_eye = cv2.resize(r_eye,(24,24))
                r_eye = r_eye/255
                r_eye =  r_eye.reshape(24,24,-1)
                r_eye = np.expand_dims(r_eye,axis=0)
                rpred = model.predict(r_eye)
                rpred=np.argmax(rpred)
                    #print(rpred)
                if(rpred==1):
                    lbl='Open' 
                if(rpred==0):
                    lbl='Closed'
                

            for (x,y,w,h) in left_eye:
                l_eye=frame[y:y+h,x:x+w]
                count=count+1
                l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)  
                l_eye = cv2.resize(l_eye,(24,24))
                l_eye= l_eye/255
                l_eye=l_eye.reshape(24,24,-1)
                l_eye = np.expand_dims(l_eye,axis=0)
                lpred = model.predict(l_eye)
                lpred=np.argmax(lpred)
    
                if(lpred==1):
                    lbl='Open'   
                if(lpred==0):
                    lbl='Closed'
                

            if(rpred==0 and lpred==0):
                score=score+1
                cv2.putText(frame,"Closed",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
            # if(rpred[0]==1 or lpred[0]==1):
            else:
                score=score-1
                cv2.putText(frame,"Open",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)

    
            if(score<0):
                score=0   
            cv2.putText(frame,'Score:'+str(score),(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
            if(score>20):
                #person is feeling sleepy so we beep the alarm
                #cv2.imwrite(os.path.join(path,'image.jpg'),frame)
                try:
                    sound.play()
                    cv2.putText(frame,"Drowsy",(100,100),font,1,(225,250,255),1,cv2.LINE_AA)
        
                except:  # isplaying = False
                    pass
                if(thicc<21):
                    thicc= thicc+2
                else:
                    thicc=thicc-2
                    if(thicc<2):
                        thicc=2
                cv2.rectangle(frame,(0,0),(width,height),(0,0,255),thicc) 
    
    

            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()
            
            yield(b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')    



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(get_frames(cam,leye,reye,count,score,thicc,font,face,path,rpred,lpred),mimetype='multipart/x-mixed-replace; boundary=frame') 

if __name__=='__main__':
    app.run(debug=True)

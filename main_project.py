from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
import os
import matplotlib as plt

face_classifier = cv2.CascadeClassifier(r'.\haarcascade_frontalface_default.xml')
classifier =load_model(r'.\model.h5')
dataset_path=os.getcwd()+"\\dataset"
label_emotion = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']
#face_model_path=os.getcwd()+"\\face_detector" 
#---------------------------------------------------------------------------------------------
def create_dataset_folders(dataset_path,labels):
    for label in labels:
        dataset_folder = dataset_path+"\\"+label
        if not os.path.exists(dataset_folder):
            os.makedirs(dataset_folder)

#------------------------------------------------------------------------------------------------------
def color_modification(im,label,dataset_path):
   # im=np.asarray(im)
    #a=np.random.randint(1,6)
    #b=np.random.randint(1,5)
    #res=a*im + b
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    #res = cv2.equalizeHist(img)
    res = cv2.calcHist([img],[0],None,[256],[0,256])
    capture_face_expression(res,label,dataset_path)



#---------------------------------------------------------------------------------------------
def capture_face_expression(face_expression,label,dataset_path):
    if len(face_expression)!=0:
        try :
            dataset_folder = dataset_path+"\\"+label
            number_files = len(os.listdir(dataset_folder)) # dir is your directory path
            new_folder = dataset_folder+"\\"+str(number_files)
            if not os.path.exists(new_folder):
                 os.makedirs(new_folder)  
            image_path  = "%s\\%s_%d.png"%(new_folder,label,number_files)      
            cv2.imwrite(image_path, face_expression)
            
        except :
            print("ERREUR lors de l'enregistrement du capture")
#---------------------------------------------------------------------------------------------
print("[INFO] Creating dataset folders...")
create_dataset_folders(dataset_path,label_emotion)
cap = cv2.VideoCapture(0)



while True:
    _, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray1 = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)



        if np.sum([roi_gray1])!=0:
            roi = roi_gray1.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

            prediction = classifier.predict(roi)[0]
            label=label_emotion[prediction.argmax()]
            label_position = (x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            
            if cv2.waitKey(1) & 0xFF == ord('c'):
                print("Key C pressed")
                _, frame1 = cap.read()
                #face_captured=cv2.resize(roi_gray,(224,224),interpolation=cv2.INTER_AREA)
                color_modification(frame1,label,dataset_path)
        else:
            cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv2.imshow('Emotion Detector',frame)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        print("Key S pressed")
        break

cap.release()
cv2.destroyAllWindows()
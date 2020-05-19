import cv2
cap= cv2.VideoCapture(0)

while(True):
    
    ret,frame= cap.read()
    face_classifier=cv2.CascadeClassifier('/Users/pk/Desktop/Python for DS course/haarcascade_frontalface_alt.xml')
    eye_identifier=cv2.CascadeClassifier('/Users/pk/Desktop/Python for DS course/haarcascade_eye.xml')
    if ret==False:
        continue
    
    facefeatures= face_classifier.detectMultiScale(frame,1.3,5)
    eyes= eye_identifier.detectMultiScale(frame,1.3,3)
    
    for (x,y,h,w) in eyes:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        
    for (x,y,h,w) in facefeatures:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
    
    cv2.imshow('video',frame)
    
    key_pressed= cv2.waitKey(1) & 0xff
    if key_pressed==ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
    
import cv2

haar_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

cam = cv2.VideoCapture(0) #initialize camera

while True:
    ret,img = cam.read() # Redaing from camera
    
    grayImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # Converting color
    
    faces = haar_cascade.detectMultiScale(grayImg, 1.3, 4) #Getting face

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)

    cv2.imshow("FaceDetection", img) # display the frame

    key = cv2.waitKey(10)
    print(key)
    
    if key==27:
        break

cam.release()
cv2.destroyAllWindows()

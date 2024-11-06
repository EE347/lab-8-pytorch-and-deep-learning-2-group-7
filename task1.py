import cv2 
from picamera2 import Picamera2, MappedArray
from datetime import datetime
import numpy as np

cam = Picamera2()
cam.configure(cam.create_preview_configuration(main={"size": (1920, 1080)}))
cam.start() 

width = 1920
height = 1080

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

recording = False
face = False

picture_index = 0
video_index = 0

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

output = None

target_size = (64, 64)

while True: 
    
    frame = cv2.cvtColor(cam.capture_array(), cv2.COLOR_RGB2BGR)

    if recording:
        if face:
            grey_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(grey_image, scaleFactor=1.1, minNeighbors=5, minSize=(40,40))
            for (x, y, w, h) in faces:
                frame = cv2.rectangle(grey_image, (x,y), (x+w, y+h), (0,255,0), 4)
                cropped_face = frame[y:y+h, x:x+w]
                if cropped_face is not None:
                    output.write(cropped_face)
        frame = cv2.circle(frame, (20, 20), 5, (0, 0, 255), 10)
        cv2.imshow("Live", frame)
    else:
        cv2.imshow("Live", frame)
    
    key = cv2.waitKey(1)
    if key == ord("p"):
        pic_time = datetime.now().strftime("%d%m%Y%H%M%S")
        pic_name = "image" + str(pic_time) + ".jpg"
        if face:
            grey_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(grey_image, scaleFactor=1.1, minNeighbors=5, minSize=(40,40))
            for (x, y, w, h) in faces:
                face_image = cv2.rectangle(grey_image, (x,y), (x+w, y+h), (0,255,0), 4)
                cropped_face = frame[y:y+h, x:x+w]
                if cropped_face is not None:
                    cropped_face = cv2.resize(cropped_face, target_size, interpolation=cv2.INTER_AREA)
                    cv2.imwrite("face_" + pic_name, cropped_face)
        else:
            cv2.imwrite(pic_name, frame)


    if key == ord("v"):
        if output is not None:
            output.release()
        recording = not recording
        if recording:
            vid_time = datetime.now().strftime("%d%m%Y%H%M%S")
            vid_name = "video" + str(vid_time) + ".mp4"
            output = cv2.VideoWriter( vid_name,fourcc, 20.0, (width,height))

    if key == ord("f"):
        face = not face

    if key == ord("q"):
        if output is not None:
            output.release()
        break

# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 13:17:04 2020

@author: Aviral singh halsi
"""
import cv2
import numpy as np

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def face_extractor(img):
    faces = face_classifier.detectMultiScale(img, 1.3, 5)

    if faces is ():
        return None

    for (x,y,w,h) in faces:
        x=x-10
        y=y-10
        cropped_face = img[y:y+h+50, x:x+w+50]

    return cropped_face

count=0

while (count<531):
    count=count+1
    image=cv2.imread('./training/richard/jhye ('+str(count)+').jpg')
    if image is not None:
        if face_extractor(image) is not None:
            face = cv2.resize(face_extractor(image), (400, 400))
            file_name_path = './faces/' + str(count) + '.jpg'
            cv2.imwrite(file_name_path, face)
        else:
            print("face not in"+str(count))
    else:
        pass
cv2.waitKey(0)
cv2.destroyAllWindows()
print(str(count))
print("Collecting Samples Completed")

from imutils import paths
import os
import cv2
import numpy as np
from joblib import dump,load

clf = load("cat&dog.joblib")

img = cv2.imread(r"C:\Users\S A R I R\Desktop\cat2.jpg")
img1 = cv2.resize(img , (32,32))
img1 = img1/255.0
img1 = img1.flatten()

output = str (clf.predict(np.array([img1]))[0])


Dog_Class = 1
Cat_Class = 0

cv2.putText(img,output, (20,40),cv2.FONT_HERSHEY_COMPLEX,0.8,(255,0,0),2)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

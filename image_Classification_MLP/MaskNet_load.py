from keras import models
import cv2
import numpy as np
model = models.load_model(r"C:\Users\S A R I R\Desktop\MLP_mask.h5")

img = cv2.imread(r"C:\Users\S A R I R\Desktop\mask5.jpg")
img1 = cv2.resize(img , (32, 32))
img1 = img1/255.0
img1 = img1.flatten()
img1 = np.array([img1])

output = model.predict(img1)[0]
output_max = np.argmax(output)

category_name = ['no mask', 'mask']

text = "{}: {:.2f}".format(category_name[output_max], output[output_max] * 100)

cv2.putText(img, text, (20,40), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,250) ,2)
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

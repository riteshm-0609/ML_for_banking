import cv2
from cv2 import fastNlMeansDenoisingColored
import pytesseract
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
import imutils
import os

img = cv2.imread("/home/ritesh/Desktop/test2.jpg")
img = cv2.fastNlMeansDenoisingColored(img,10,1,0,7,21)
plt.imshow(img)
img1=mpimg.imread('/home/ritesh/Desktop/test2.jpg')
name = img[700:780, 480:1200]
filename = "{}.png".format(os.getpid())
cv2.imwrite(filename, name)
plt.imshow(name)
text = pytesseract.image_to_string(Image.open(filename),lang='eng')
os.remove(filename)
print(text)

import cv2
import time




def crop_image(img, res=(512, 512)):

    height, width = img.shape[:2]

    if width > height:
        img = img[0:height, int(width/2 - height/2):int(width/2 + height/2)]
    if height > width:
        img = img[int(height/2 - width/2):int(height/2 + width/2), 0:width]

    img = cv2.resize(img, res)

    return img


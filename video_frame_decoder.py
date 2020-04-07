import os

import cv2
import numpy
from PIL import Image, ImageDraw


def convert_cv2_to_PIL(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    return im_pil


def convert_PIL_to_cv2(image):
    pil_image = image.convert('RGB')
    open_cv_image = numpy.array(pil_image)
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    return open_cv_image


def mark_image(cv2_image):
    image = convert_cv2_to_PIL(cv2_image)
    draw = ImageDraw.ImageDraw(image)
    draw.rectangle([(50, 50), (100, 100)])
    return convert_PIL_to_cv2(image)


vidcap = cv2.VideoCapture('small.mp4')

fps = vidcap.get(5)
print(fps)
success, image = vidcap.read()

fourcc = cv2.VideoWriter_fourcc(*'MP4V')
height, width, channels = image.shape
out = cv2.VideoWriter('output.mp4',fourcc, fps ,(width,height),True )
count = 0
while success:
    out.write(mark_image(image))
    success, image = vidcap.read()
    print('Read a new frame: ', success)
    count += 1

cv2.destroyAllWindows()
out.release()
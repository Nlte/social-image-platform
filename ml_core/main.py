from PIL import Image
from image_processing import process_image
import numpy as np
import tensorflow as tf

sess = tf.InteractiveSession()

im = Image.open('landscapetree.jpg')
with open('landscapetree.jpg') as f:
    im_str = f.read()
image = process_image(im_str, 299, 299).eval()
crop_img = Image.fromarray(image)
crop_img.show()

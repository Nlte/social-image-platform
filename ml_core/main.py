import tensorflow as tf

import image_processing
import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

with open("im24707.jpg") as f:
    img_data = f.read()


original = image_processing.process_image(img_data, 299, 299)
resized_img = image_processing.process_image(img_data, 299, 299, distort=True)

sess = tf.InteractiveSession()
myimg = sess.run(resized_img)
originalimg = sess.run(original)

fig = plt.figure()
# original
a=fig.add_subplot(1,2,1)
imgplot = plt.imshow(originalimg)
a.set_title('Original')
# distort
a=fig.add_subplot(1,2,2)
imgplot = plt.imshow(myimg)
a.set_title('Distorted')

plt.show()

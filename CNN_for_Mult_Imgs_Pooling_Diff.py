from os import listdir
from pathlib import Path
from os.path import isfile, join
from keras.layers import Convolution2D, MaxPooling2D, Activation
from keras.models import Sequential
from numpy import asarray
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
path=Path('D:/Neurons/Images')
imagepath='D:/Neurons/Images'
images = []
#onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath,f))]
#img = np.empty(len(onlyfiles), dtype=object)
#for n in range(0, len(onlyfiles)):
for imagepath in path.glob("*.jpg"):
    #img[n] = cv2.imread( join(mypath,onlyfiles[n]))
    #img = cv2.imread('Tesla.jpg', cv2.IMREAD_GRAYSCALE) #convert to gray scale
    img = cv2.imread(str(imagepath))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img,(512,350))
    images.append(img)
    #img = img
    plt.imshow(img, cmap = 'gray')
    #img.shape = (512, 512)
    fig = plt.gcf()
    fig.canvas.set_window_title('Image'+str(imagepath))
    #plt.show()
    plt.show(block = False)
    plt.pause(1)
    plt.close()
    #img.shape[0] = 512
    #img.shape[1] = 512

    print(img.shape)#resolution of image
    img_batch = img.reshape(1, img.shape[0], img.shape[1], 1)
    print(img_batch.shape)

    model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(1, (15,15),
            padding = 'valid', input_shape = img_batch.shape[1:])])#random wts
    #model.summary()#model summary
    #print(model1.summary)
    conv_image = model.predict(img_batch)
    print(conv_image.shape)#new shape after convolution
    #features are highlighted and retained so that processing is lesser
    #in the next steps and better
    conv_img_show = conv_image.reshape(conv_image.shape[1], conv_image.shape[2])#only 1 photo and that too in greyscale 
    print(conv_img_show)#no 1st and 4th arguments because they already have been 
    plt.imshow(conv_img_show, cmap = 'gray')#consolidated 
    fig2 = plt.gcf()
    fig2.canvas.set_window_title('Conv Image')
    plt.show(block = False)
    plt.pause(1)
    plt.close()


    model1 = tf.keras.models.Sequential([tf.keras.layers.Conv2D(1, (15,15),
             padding = 'valid', activation = 'relu',
             input_shape = img_batch.shape[1:])])

    conv_img1 = model1.predict(img_batch)
    conv_img1_show = conv_img1.reshape(conv_img1.shape[1], conv_img1.shape[2])
    plt.imshow(conv_img1_show, cmap = 'gray')
    fig3 = plt.gcf()
    fig3.canvas.set_window_title('Conv with ReLU')#features further highlighted
    plt.show(block = False)
    plt.pause(1)
    plt.close()


    model2 = tf.keras.models.Sequential([tf.keras.layers.Conv2D(1, (15,15),
             padding = 'valid', activation = 'relu',
             input_shape = img_batch.shape[1:]),
             tf.keras.layers.MaxPool2D(2,2)])

    conv_img2 = model1.predict(img_batch)
    conv_img2_show = conv_img1.reshape(conv_img2.shape[1], conv_img2.shape[2])
    print("Size after pooling:")
    print(conv_img2.shape)
    plt.imshow(conv_img2_show, cmap = 'gray')

    fig3 = plt.gcf()
    fig3.canvas.set_window_title('Pool after Conv')
    plt.show(block = False)
    plt.pause(1)
    plt.close()
    #model2.summary()





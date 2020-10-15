#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import matplotlib.pyplot as plt
#import seaborn as sns
import numpy as np
from PIL import ImageTk, Image, ImageDraw
import PIL
from tkinter import *
import cv2
print("tf version : ",tf.__version__)
print("numpy version:",np.__version__)
print("cv2 version:",cv2.__version__)


# In[2]:


def drawing():
    model = tf.keras.models.load_model("./aug_digit.h5")
    width = 300
    height = 300
    center = height//2
    white = (255, 255, 255)
    green = (0,128,0)
    
    
    def predict():
        img=cv2.imread('image.png',0)
        img=cv2.bitwise_not(img)

        img=cv2.resize(img,(28,28))
        img=img.reshape(1,28,28,1)
        img=img.astype('float32')
        img=img/255.0
        #print(img)
        pred=model.predict(img)
        myLabel.config(text = "Predicted digit : "+str(pred[0].argmax()))
        
        #_ , ax = plt.subplots(1,2,figsize = (12 , 6))
        #ax[0].imshow(img[0,:,:,0])
        #ax[0].set_title("Predicted digit : "+str(pred[0].argmax()))
        #plt.ioff()
        #plt.bar(list(range(0,10)) , pred[0])
        #plt.xticks(list(range(0,10)))
        #plt.imshow(img[0,:,:,0])
        #plt.show()
        
        #plt.ioff()
        plt.bar(list(range(0,10)) , pred[0])
        plt.xticks(list(range(0,10)))
        plt.show()
        
        
    def save():
        filename = "image.png"
        image1.save(filename)
        predict()
        

    def paint(event):
        x1, y1 = (event.x - 1.5), (event.y - 1.5)
        x2, y2 = (event.x + 1.5), (event.y + 1.5)
        cv.create_oval(x1, y1, x2, y2, fill="black",width=30)
        draw.line([x1, y1, x2, y2],fill="black",width=30)
        
    """
    def clear():
        cv.delete("all")
        myLabel.config(text = "Predicted digit : ")
        plt.close(1)
    """
    
    root = Tk()
    root.title("MNIST Digit Prediction")
    myLabel = Label(root , text = "Predicted digit : " , font = "Times 12 bold")
    myLabel.pack()
    cv = Canvas(root, width=width, height=height, bg='white')
    cv.pack()
    
    image1 = PIL.Image.new("RGB", (width, height), white)
    draw = ImageDraw.Draw(image1)
    cv.pack(expand=YES, fill=BOTH)
    cv.bind("<B1-Motion>", paint)
    
    btn1 = Button(root , text = "predict",width=42,pady = 10 , command = save)
    btn1.pack()
    
    #btn2 = Button(root , text = "Clear",width=42,pady = 10 , command = clear)
    #btn2.pack()
    
    root.mainloop()


# In[ ]:


drawing()


# In[ ]:





import cv2
import glob
import math
import numpy as np
import os.path
from keras.models import Model
from keras.layers import Conv2D, Input, Add
from keras.optimizers import Adam

def create_LR(src,scale):
    LR=[]
    for i in range(len(src)):
        label_=src[i]
        h,w,_=label_.shape
        input_=cv2.resize(label_,(round(w/scale),round(h/scale)),interpolation=cv2.INTER_CUBIC)
        input_=cv2.resize(input_,(w,h),interpolation=cv2.INTER_CUBIC) 
        LR.append(input_)
    return LR

def loadimg(path):
    files = glob.glob(path)
    src=[] 
    for file in files:
        src.append(cv2.imread(file,1)/255.0)
    return src

def subimg(src,r_field):
    sub_gt=[]
    for i in range(len(src)):
        h,w,_=src[i].shape
        h=h-h%r_field
        w=w-w%r_field
        t=src[i]
        for a in range(0,h,r_field):
            for b in range(0,w,r_field):
                tt=t[a:a+r_field, b:b+r_field]
                if(tt.shape==(r_field,r_field,3)):
                    sub_gt.append(tt.reshape(r_field,r_field,3))
    sub_gtt=np.array(sub_gt) 
    return sub_gtt

def vdsr_model_add(r_field):
    x_in=Input(shape=(r_field,r_field,3),name='input')
    x=Conv2D(64,kernel_size=3,padding='same',activation='relu')(x_in)
    for i in range(19):
        x=Conv2D(64,kernel_size=3,padding='same',activation='relu')(x)
    x_out=Conv2D(3,kernel_size=3,padding='same',name='output')(x)
    add=Add(name='add')([x_in, x_out])
    model=Model(x_in,add)
    return model

def reconstruction(sub_image,H,W,img_size):
    H = H - np.mod(H, img_size)
    W = W - np.mod(W, img_size)
    recon=np.zeros((H,W,3))
    count=0
    for x in range(0,H-img_size+1,img_size):
        for y in range(0,W-img_size+1,img_size):
            recon[x:x+img_size, y:y+img_size]=sub_image[count]
            count+=1
    return recon

def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    return format(20 * math.log10(PIXEL_MAX / math.sqrt(mse)),'.3f')

def parentDir():
    curDir=os.path.abspath(os.path.join(__file__, os.pardir))
    parentDir=os.path.abspath(os.path.join(curDir, os.pardir))
    _parentDir=parentDir+"/"
    return _parentDir


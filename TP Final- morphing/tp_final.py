# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 13:49:43 2022

@author: djouad
"""

import numpy as np
from skimage.io import imread
from skimage.io import imshow,imsave
from matplotlib import pyplot as plt
from skimage.color import rgb2gray
from skimage.color import rgba2rgb
from skimage import img_as_ubyte
from skimage.morphology import dilation, erosion, disk, opening
import cv2
import PIL
import imageio
def pil_to_cv(pil_image):
    """
    Returns a copy of an image in a representation suited for OpenCV
    :param pil_image: PIL.Image object
    :return: Numpy array compatible with OpenCV
    """
    return np.array(pil_image)[:, :, ::-1]

# Translation d'image
def translation_img(src_img,shift_distance, shape_of_out_img):
    h,w = src_img.shape[:2] 
    out_img=np.zeros(shape_of_out_img,src_img.dtype)
    for i in range(h):
        for j in range(w):
            new_x=j+shift_distance[0] 
            new_y=i+shift_distance[1] 
            if 0<=new_x<w and 0<=new_y<h: # Test d'écriture
                out_img[new_y,new_x]=src_img[i,j]
    return out_img
    
    
    # ENTREE = FORMES EN NOIR SUR FOND BLANC
    # INDIQUER LES NOMS DE VOS PROPRES FICHIERS ICI

IN1=imread('i1.png',as_gray=True)
IN2=imread('i2.png',as_gray=True)

# Conversion en images binaires GRAY -> BOOLEAN
IM1=(IN1==0)
imshow(np.uint8(IM1)*255) # Pour la visu, images 8 bits
plt.show()

IM2=(IN2==0)
imshow(np.uint8(IM2)*255)
plt.show()


# Positionnement / Centrage des 2 formes

# centre de l'image
(szx,szy)=IM1.shape
centrex=int(szx/2)
centrey=int(szy/2)

# centrage Forme 1

(tabx, taby)=np.where(IM1==1)
mx=int(np.sum(tabx)/np.size(tabx))
my=int(np.sum(taby)/np.size(taby))
dx=int(centrex-mx)
dy=int(centrey-my)

FORME1=translation_img(IM1,(dy,dx), IM1.shape)
  
        

# centrage Forme 2
(tabx, taby)=np.where(IM2==1)
mx=np.rint(np.sum(tabx)/np.size(tabx))
my=np.rint(np.sum(taby)/np.size(taby))
dx=int(centrex-mx)
dy=int(centrey-my)


FORME2=translation_img(IM2,(dy,dx), IM2.shape)


#imshow(np.uint8(FORME1)*255) # Pour la visu, images 8 bits
plt.show()
#imshow(np.uint8(FORME2)*255) # Pour la visu, images 8 bits
plt.show()

INTER = np.minimum((FORME1),(FORME2)) 

#imshow(np.uint8(INTER)*255)
UNION = np.maximum ((FORME1),(FORME2))
#imshow(np.uint8(UNION)*255)
E=INTER
TMP=INTER
i=1



def mediane(F1,F2) :
 INTER= np.minimum((F1),(F2)) 
 UNION = np.maximum(F1,F2)
 E= INTER
 i=1
 TMP = INTER

 
 
 while np.max(TMP>0) :
 
  TMP=erosion(UNION,disk(i))
  #print(TMP)
  E=np.maximum(E,np.minimum(TMP,dilation(INTER,disk(i))))
  i=i+1
  
 '''# imshow(E,cmap=plt.cm.gray)
  plt.title('Image erosion')
  plt.show()'''
 return E




m=mediane(FORME1,FORME2)

Liste_mediane =[FORME1,m,FORME2]


n=8 # 2^n+1 images
j= 0
for k in range(n):
    print(k)
    nb=0
    longueur=len(Liste_mediane)-1
    for j in range (longueur) :
        me =mediane(Liste_mediane[j+nb],Liste_mediane[j+nb+1])
        imshow(me,cmap=plt.cm.gray)
        plt.show()
        Liste_mediane.insert(j+1+nb,me) 
        
        nb+=1

    
Liste_mediane2 = []    
for v in range(len(Liste_mediane)) :
 Liste_mediane2.append(np.uint8((Liste_mediane[v])*255))

imageio.mimsave('vid.gif',Liste_mediane2,fps=20)
'''on peut pas utiliser cette partie du programme car ca marche uniquement sur des images
    en RVB dimensions "en couleurs" mais dans le cas on des images en '''


# def write_video(file_path, frames, fps):
#     """
#     Writes frames to an mp4 video file
#     :param file_path: Path to output video, must end with .mp4
#     :param frames: List of PIL.Image objects
#     :param fps: Desired frame rate
#     """

#     w, h = frames[0].shape
#     fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
#     writer = cv2.VideoWriter(file_path, fourcc, fps, (w, h))

#     for frame in frames:
#         writer.write(PIL.Image(frame))

#     writer.release()
# '''

#ovrire -> lire ->ecrire ->fermer 
# for k in range(len(Liste_mediane)) : 
    
#  fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')   
#  out = cv2.VideoWriter('vid.mp4', k, 20.0, (112, 200))
#  out.write(k)

#  out.release()
 
# from PIL import Image, ImageSequence
# import sys, os
# filename = sys.argv[0]
# im = Image.open(filename)
# original_duration = im.info['duration']
# frames = [frame.copy() for frame in ImageSequence.Iterator(im)]    
# frames.reverse()

# from images2gif import writeGif
# writeGif("reverse_" + os.path.basename(filename), Liste_mediane2, duration=original_duration/1000.0, dither=0)
# '''
# write_video("out.mp4", Liste_mediane, 15)

# # def union(I1,I2):
# #     imageout=I1
# #     for x in range (0,I1.shape[0]):
# #         for y in range (0,I1.shape[1]):   
# #             imageout[x][y]=max(I1[x][y],I2[x][y])                  
# #     return imageout

# # IMG4= union(FORME1, FORME2)
# # imshow(IMG4,cmap=plt.cm.gray)
# # plt.title('Image Union')
# # plt.show()

# # def inter(I1,I2):
# #     imageout=I1
# #     for x in range (0,I1.shape[0]):
# #         for y in range (0,I1.shape[1]):   
# #             imageout[x][y]=min(I1[x][y],I2[x][y])                  
# #     return imageout
# # IMG5= inter(FORME1, FORME1)
# # imshow(IMG5,cmap=plt.cm.gray)
# # plt.title('Image infinimuim')
# # plt.show()
# '''
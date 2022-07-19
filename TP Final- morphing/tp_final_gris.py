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
from PIL import Image

#charger image 1 

IN1=imread('img1.jpg')

#charger image 2
IN2=imread('img2.jpg')

def mediane(F1,F2) :
 INTER= np.minimum((F1),(F2)) 
 UNION = np.maximum(F1,F2)
 E= INTER
 i=1
 TMP = INTER

 
 
 while np.max(TMP!=0) :
 
  TMP=erosion(UNION,disk(i))
  #print(TMP)
  E=np.maximum(E,np.minimum(TMP,dilation(INTER,disk(i))))
  i=i+i
  print(i)
  
  imshow(E,cmap=plt.cm.gray)
  plt.title('Image erosion')
  plt.show()
 return E


FORME1=IN1[:,:,0]
FORME2=IN2[:,:,0]

m=mediane(FORME1,FORME2)

Liste_mediane =[FORME1,m,FORME2]


n=2 # 2^n+1 images
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
import skimage
import numpy as np
from skimage import io
from skimage.io import imread
from skimage.io import imshow
from matplotlib import pyplot as plt
from skimage import filters
from skimage.color import rgb2gray
from skimage.color import rgba2rgb
from skimage import img_as_ubyte
from skimage.morphology import dilation
from skimage.morphology import erosion
from skimage.morphology import disk
from skimage.morphology import opening
import skimage.morphology as morpho  


def strel(forme,taille,angle=45):
    """renvoie un element structurant de forme  
     'diamond'  boule de la norme 1 fermee de rayon taille
     'disk'     boule de la norme 2 fermee de rayon taille
     'square'   carre de cote taille (il vaut mieux utiliser taille=impair)
     'line'     segment de langueur taille et d'orientation angle (entre 0 et 180 en degres)
      (Cette fonction n'est pas standard dans python)
    """

    if forme == 'diamond':
        return morpho.selem.diamond(taille)
    if forme == 'disk':
        return morpho.selem.disk(taille)
    if forme == 'square':
        return morpho.selem.square(taille)
    if forme == 'line':
        angle=int(-np.round(angle))
        angle=angle%180
        angle=np.float32(angle)/180.0*np.pi
        x=int(np.round(np.cos(angle)*taille))
        y=int(np.round(np.sin(angle)*taille))
        if x*2+y*2 == 0:
            if abs(np.cos(angle))>abs(np.sin(angle)):
                x=int(np.sign(np.cos(angle)))
                y=0
            else:
                y=int(np.sign(np.sin(angle)))
                x=0
        rr,cc=morpho.selem.draw.line(0,0,y,x)
        rr=rr-rr.min()
        cc=cc-cc.min()
        img=np.zeros((rr.max()+1,cc.max()+1) )
        img[rr,cc]=1
        return img
    raise RuntimeError('Erreur dans fonction strel: forme incomprise')



IN1=imread('rice.png')
GRAY1=IN1;
imshow(GRAY1,cmap=plt.cm.gray)
plt.title('Image Originale du rice')
plt.show()

Rmax = 30
VOL =np.zeros(Rmax)
VOL[0]=np.sum(GRAY1)
for i in range(1,Rmax):
    B= opening(IN1,strel('line',i,30))
    VOL[i]=np.sum(B)
    
    plt.plot(VOL)
    plt.title('Courbe de la décroissance du rayon en fonction du volume')
    plt.show()
    
dVOL=np.zeros(Rmax-1)
for i in range(Rmax-2):
    dVOL[i] = VOL[i] - VOL[i+1]
    
    plt.plot(dVOL)
    plt.title('Courbe de la décroissance du rayon en fonction du volume')
    plt.title('Volume de l"image')
    plt.title('Dérivée')
    plt.show()
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

#Importation de l’image
img = imread('rice.png') ;
GRAY=img; 
N = 4 # Nombre d'iteration
 
'''imgcv=im2bw(img); #convertit l’image en niveaux de gris en image binaire 
figure(N+1); # affichage de toutes les figures
imshow(imgcv);
title('image de riz en binaire ');'''

seuil=skimage.filters.threshold_otsu(GRAY)
bin=GRAY>=seuil
B=img_as_ubyte(bin)
imshow(B)
plt.title('Image Binaire')
plt.show()

sizemat = np.shape(B)
matrice = np.zeros(sizemat)
print(matrice)
for i in range(1,N):
    # im = strel('disk',i)
    # erosion = imerode(imgcv,im);
     E=erosion(B,disk(i))
     imshow(E)
     plt.title('Image Erodée')
     plt.show()
'''
     if sum(sum(E))>0 :
       # seq1t = strel('disk',2);
       # ouverture = imopen(erosion,seq1t);
        E2=erosion(B,disk(2))
        ouverture = imopen(E,E2);
       
#mettre les composantes du squelette dans la matrice 
#couverture de lérosion de cette dernière
        cou = double(E) - double(ouverture);
        mat[:,:,i+1] = cou;
        matrice = matrice + double(mat[:,:,i+1]);
'''
    
'''cou = double(erosion) - double(ouverture);
    mat(:,:,i+1) = cou;
    matrice = matrice + double(mat(:,:,i+1));
        
        figure(i+1) 
        imshow(cou);
     else
         i=N
figure(3)
imshow(matrice);
title('matrice de squelette')'''
